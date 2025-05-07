#!/usr/bin/env python3
# contrastive_pretrain.py

import os
import ast
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR        = "../datasets/interim/embeddings/pt"
SBERT_FILE      = "sbert_output_1.pt"
ORIG_CSV        = "../datasets/interim/translated_output_1.csv"
CLUSTER_CSV     = "../results/contrastive_analysis/cooccurrence.csv"
PRETRAIN_EPOCHS = 10
BATCH_SIZE      = 128
LR              = 1e-3
MARGIN          = 0.5
OUTPUT_PROJ     = "../results/contrastive_analysis/sbert_contrastive.npy"
os.makedirs(os.path.dirname(OUTPUT_PROJ), exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# ─── 1. LOAD EMBEDDINGS, LABELS & CLUSTERS ────────────────────────────────────
def load_data():
    # labels
    df0 = pd.read_csv(ORIG_CSV)
    labels = []
    for x in df0["CommentClass_en"]:
        if isinstance(x, str) and x.startswith("["):
            try: lbl = ast.literal_eval(x)
            except: lbl = [x]
        else:
            lbl = [x]
        labels.append(lbl)
    # multi‑hot
    stacked = pd.DataFrame(labels).stack()
    dummies = pd.get_dummies(stacked, dtype=int)
    y = dummies.groupby(level=0).sum().values  # (N, C)

    # embeddings
    data = torch.load(os.path.join(DATA_DIR, SBERT_FILE))
    feats = data.get("features", data.get("embeddings"))
    X = feats.cpu().numpy() if isinstance(feats, torch.Tensor) else np.array(feats)

    # clusters: re‑compute KMeans to match what you used
    # (alternatively load from file if you saved labels)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    clusters = km.labels_

    return X, y, clusters

# ─── 2. TRIPLET DATASET BASED ON CLUSTERS & LABELS ────────────────────────────
class TripletDataset(Dataset):
    def __init__(self, X, y, clusters):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.N = len(X)
        # build index maps
        self.by_cluster = {}
        self.by_label = {}
        for i in range(self.N):
            c = clusters[i]
            self.by_cluster.setdefault(c, []).append(i)
            for lbl in np.where(y[i] == 1)[0]:
                self.by_label.setdefault(lbl, []).append(i)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        anchor = self.X[idx]
        c = self.clusters[idx]
        pos_idx = idx
        # positive: same cluster AND share at least one label
        anchor_lbls = np.where(self.y[idx] == 1)[0]
        # try to find index in same cluster sharing a label
        candidates = []
        for lbl in anchor_lbls:
            candidates += [i for i in self.by_label[lbl] if self.clusters[i] == c and i != idx]
        if candidates:
            pos_idx = random.choice(candidates)
        # negative: different cluster AND share no labels
        neg_candidates = [i for i in range(self.N)
                          if self.clusters[i] != c and not (self.y[i] & self.y[idx]).any()]
        neg_idx = random.choice(neg_candidates) if neg_candidates else random.randrange(self.N)

        return (
            torch.from_numpy(anchor).float(),
            torch.from_numpy(self.X[pos_idx]).float(),
            torch.from_numpy(self.X[neg_idx]).float()
        )

# ─── 3. PROJECTION HEAD ────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

# ─── 4. TRAIN CONTRASTIVE PROJECTION ───────────────────────────────────────────
def train_contrastive(X, y, clusters):
    dataset = TripletDataset(X, y, clusters)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProjectionHead(input_dim=X.shape[1], proj_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    model.train()
    for epoch in range(1, PRETRAIN_EPOCHS+1):
        total = 0.0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            za, zp, zn = model(a), model(p), model(n)
            loss = criterion(za, zp, zn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * a.size(0)
        logging.info(f"Epoch {epoch}/{PRETRAIN_EPOCHS}  Avg TripletLoss = {total/len(dataset):.4f}")

    # produce new embeddings
    model.eval()
    with torch.no_grad():
        Z = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    np.save(OUTPUT_PROJ, Z)
    logging.info(f"Saved contrastively‑trained embeddings to {OUTPUT_PROJ}")
    return Z

# ─── 5. EVALUATE CLUSTER–LABEL ALIGNMENT PRE vs. POST ──────────────────────────
def eval_alignment(Z, y, clusters):
    # re‑cluster Z
    k = len(np.unique(clusters))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    labels_z = km.labels_
    flat = np.argmax(y, axis=1)
    nmi = normalized_mutual_info_score(flat, labels_z)
    logging.info(f"Post‑contrastive NMI = {nmi:.4f}")

def main():
    X, y, clusters = load_data()
    logging.info("Starting contrastive pre‑training …")
    Z = train_contrastive(X, y, clusters)
    logging.info("Evaluating cluster–label alignment before and after:")
    eval_alignment(X, y, clusters)   # before
    eval_alignment(Z, y, clusters)   # after

if __name__ == "__main__":
    main()
