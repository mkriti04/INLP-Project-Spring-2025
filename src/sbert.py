import pandas as pd
import torch
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import numpy as np
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# df = pd.read_csv("./datasets/interim/translated_output_1.csv")  # Replace with your CSV filename
df = pd.read_csv("../datasets/interim/translated_output_1_clean.csv")  # Replace with your CSV filename

# Preprocess text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['Comment_en'].astype(str).apply(preprocess_text)
df['CommentClass_en'] = df['CommentClass_en'].apply(eval)
df['label'] = df['CommentClass_en'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")


import torch

def compute_overlap_weights(batch_labels):
    # batch_labels: List[List[str]] of length B
    B = len(batch_labels)
    weights = torch.zeros(B, B)
    for i in range(B):
        Li = set(batch_labels[i])
        for j in range(B):
            Lj = set(batch_labels[j])
            inter = len(Li & Lj)
            union = len(Li | Lj)
            weights[i, j] = inter / union if union > 0 else 0.0
    weights.requires_grad = True
    return weights  # shape (B, B)

import torch.nn.functional as F
from torch import nn

class MultiLabelSupConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings, batch_labels):
        """
        embeddings: torch.Tensor of shape (B, D)
        batch_labels: List[List[str]] of length B
        """
        B, D = embeddings.shape
        # 1) normalize embeddings
        z = F.normalize(embeddings, dim=1)
        # 2) compute cosine similarity matrix
        sim = torch.mm(z, z.T) / self.tau
        # 3) for numerical stability subtract max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        # 4) mask out self-similarity
        mask = torch.eye(B, device=embeddings.device).bool()
        logits_masked = logits.masked_fill(mask, -1e9)
        # 5) compute label-overlap weights
        w = compute_overlap_weights(batch_labels).to(embeddings.device)
        # zero-out self-weights
        w = w.masked_fill(mask, 0.0)
        # 6) compute log-softmax
        log_prob = F.log_softmax(logits_masked, dim=1)
        # 7) weighted sum
        loss = - (w * log_prob).sum(dim=1) / (w.sum(dim=1) + 1e-8)
        return loss.mean()

train_examples = [
    (text, labels) for text, labels in zip(df['cleaned_text'], df['CommentClass_en'])
]
# custom collate to produce batch_texts, batch_labels
def collate_fn(batch):
    texts, labels = zip(*batch)
    # embeddings = model.tokenize(list(texts))
    formatted_texts = [{'text': text} for text in texts]
    return formatted_texts, list(labels)
    # return list(texts), list(labels)

train_loader = DataLoader(train_examples, shuffle=True, batch_size=32,
                          collate_fn=collate_fn)

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
criterion = MultiLabelSupConLoss(temperature=0.05).to(device)

# A simple training loop
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        embeddings, batch_labels = batch
        # SBERT’s forward to get sentence embeddings
        # z = model.encode(embeddings, convert_to_tensor=True, device=device)
        features = model.tokenize(embeddings)
        features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
        z = model(features, convert_to_tensor=True)['sentence_embedding']  # ← returns embeddings with gradient support
        loss = criterion(z, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")

model.save("sbert_mulsupcon")

# 1. Switch model to eval mode (optional but recommended)
model.eval()

# 2. Encode all texts in one batch (or in smaller batches if your data is large)
all_texts = df['cleaned_text'].tolist()
all_embeddings = model.encode(all_texts, convert_to_tensor=True, device=device)  # shape (N, D)
all_embeddings = all_embeddings.cpu()

# 3. Prepare labels and indices
all_labels = df['CommentClass_en'].tolist()  # List[List[str]]
all_indices = df.index.tolist()              # original row indices

# 4. Bundle into a dictionary
data_bundle = {
    'embeddings': all_embeddings,  # torch.Tensor
    'labels': all_labels,        # List of label-lists
    'indices': all_indices         # List of ints
}

# 5. Save to a .pt file
torch.save(data_bundle, '../datasets/interim/embeddings/pt/sbert_output_1.pt')
print("Saved embeddings, labels, and indices to 'sbert_output_1.pt'")

# 6. Convert embeddings to numpy
embedding_array = all_embeddings.numpy()

# 7. Flatten labels to semicolon-separated strings
flattened_labels = [';'.join(label_list) for label_list in all_labels]

# 8. Create column names for embeddings
embedding_cols = [f"dim_{i}" for i in range(embedding_array.shape[1])]

# 9. Create DataFrame
csv_df = pd.DataFrame(embedding_array, columns=embedding_cols)
csv_df['index'] = all_indices
csv_df['labels'] = flattened_labels

# 10. Reorder columns (optional)
csv_df = csv_df[['index', 'labels'] + embedding_cols]

# 11. Save to CSV
csv_df.to_csv('../datasets/interim/embeddings/csv/sbert_output_1.csv', index=False)
print("Saved embeddings to 'sbert_output_1.csv'")
