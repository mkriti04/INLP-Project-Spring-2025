<<<<<<< HEAD
#!/usr/bin/env python3
# contrastive_analysis.py

import os, ast, logging
from collections import Counter
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    normalized_mutual_info_score, adjusted_rand_score,
    pairwise_distances
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "../datasets/interim/embeddings/pt"
SBERT_FILE  = "sbert_output_1.pt"           # your SBERT .pt filename
ORIG_CSV    = "../datasets/interim/translated_output_1.csv"
CLASS_CSV   = "../results/classification_results.csv"  # <-- add this
RESULTS_DIR = "../results/contrastive_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# ─── 1. LOAD SBERT EMBEDDINGS & MULTI‑LABELS ───────────────────────────────────
def load_sbert():
    logging.info("Loading original labels from CSV...")
    df0 = pd.read_csv(ORIG_CSV)

    # Parse stringified lists into Python lists
    labels = []
    for x in df0["CommentClass_en"]:
        if isinstance(x, str) and x.startswith("["):
            try:
                lbl = ast.literal_eval(x)
            except:
                lbl = [x]
        else:
            lbl = [x]
        labels.append(lbl)

    # Build a multi‑hot matrix
    stacked = pd.DataFrame(labels).stack()
    dummies = pd.get_dummies(stacked, dtype=int)
    mlb = dummies.groupby(level=0).sum()   # shape = (n_samples, n_unique_labels)
    y_true = mlb.values

    logging.info("Loading SBERT embeddings from .pt file...")
    data = torch.load(os.path.join(DATA_DIR, SBERT_FILE))
    feats = data.get("features", data.get("embeddings"))
    if isinstance(feats, torch.Tensor):
        X = feats.cpu().numpy()
    else:
        X = np.array(feats)

    logging.info(f"Loaded X.shape={X.shape}, y_true.shape={y_true.shape}")
    return X, y_true, df0, mlb.columns.tolist()

# ─── 2. SELECT OPTIMAL K ────────────────────────────────────────────────────────
def select_k(X, k_min=2, k_max=10):
    sil, inert, ch = [], [], []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        lab = km.labels_
        sil.append(silhouette_score(X, lab))
        inert.append(km.inertia_)
        ch.append(calinski_harabasz_score(X, lab))
        logging.info(f"k={k}: Sil={sil[-1]:.4f}, Inertia={inert[-1]:.1f}, CH={ch[-1]:.1f}")

    # Plot metrics
    plt.figure(figsize=(15,4))
    for i,(arr,title) in enumerate(zip([sil,inert,ch], ["Silhouette","Inertia","Cal‑Har"])):
        ax = plt.subplot(1,3,i+1)
        ax.plot(ks, arr, "o-")
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "k_selection.png"))
    plt.close()

    best_k = ks[int(np.argmax(sil))]
    logging.info(f"Optimal k (max silhouette) = {best_k}")
    return best_k

# ─── 3. FINAL CLUSTERING ─────────────────────────────────────────────────────────
def cluster_data(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return km.labels_, km.cluster_centers_

# ─── 4. CLUSTER–LABEL ALIGNMENT ─────────────────────────────────────────────────
def cluster_label_alignment(y_true, labels):
    true_flat = np.argmax(y_true, axis=1)
    nmi = normalized_mutual_info_score(true_flat, labels)
    ari = adjusted_rand_score(true_flat, labels)
    logging.info(f"NMI = {nmi:.4f}, ARI = {ari:.4f}")
    with open(os.path.join(RESULTS_DIR, "alignment.txt"), "w") as f:
        f.write(f"NMI\t{nmi:.4f}\nARI\t{ari:.4f}\n")

# ─── 5. INTRA‑ VS. INTER‑LABEL VARIANCE ───────────────────────────────────────────
def variance_analysis(X, y_true, label_names):
    inv = {}
    ivr = {}
    for i, lbl in enumerate(label_names):
        idx_pos = np.where(y_true[:, i] == 1)[0]
        idx_neg = np.where(y_true[:, i] == 0)[0]
        if len(idx_pos) < 2 or len(idx_neg) < 1:
            continue
        d_pos = pairwise_distances(X[idx_pos], X[idx_pos])
        intra = d_pos[np.triu_indices_from(d_pos, k=1)].mean()
        inter = pairwise_distances(X[idx_pos], X[idx_neg]).mean()
        inv[lbl] = intra
        ivr[lbl] = inter
        logging.info(f"{lbl}: intra={intra:.4f}, inter={inter:.4f}")
    df_var = pd.DataFrame({"intra": inv, "inter": ivr})
    df_var.to_csv(os.path.join(RESULTS_DIR, "variance_analysis.csv"))
    # bar plot
    df_var.plot.bar(figsize=(10,4))
    plt.title("Intra vs Inter Label Variance")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "variance_plot.png"))
    plt.close()

# ─── 6. PROTOTYPE DISTANCE HISTOGRAMS ────────────────────────────────────────────
def prototype_distances(X, y_true, label_names):
    os.makedirs(os.path.join(RESULTS_DIR, "proto_hists"), exist_ok=True)
    for i, lbl in enumerate(label_names):
        idx_pos = np.where(y_true[:, i] == 1)[0]
        if len(idx_pos) < 2:
            continue
        proto = X[idx_pos].mean(axis=0, keepdims=True)
        d_pos = np.linalg.norm(X[idx_pos] - proto, axis=1)
        idx_neg = np.where(y_true[:, i] == 0)[0]
        d_neg = np.linalg.norm(X[idx_neg] - proto, axis=1)
        plt.figure()
        plt.hist(d_pos, bins=50, alpha=0.7, label="positive")
        plt.hist(d_neg, bins=50, alpha=0.7, label="negative")
        plt.title(f"Proto Distances for '{lbl}'")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "proto_hists", f"{lbl}.png"))
        plt.close()

# ─── 7. LABEL CO‑OCCURRENCE PER CLUSTER ─────────────────────────────────────────
def cooccurrence_by_cluster(df0, labels, y_true, label_names):
    df0c = df0.copy()
    df0c["cluster"] = labels
    rows = []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        lab_lists = [ast.literal_eval(df0c.iloc[i]["CommentClass_en"]) 
                     if isinstance(df0c.iloc[i]["CommentClass_en"], str) and df0c.iloc[i]["CommentClass_en"].startswith("[")
                     else [df0c.iloc[i]["CommentClass_en"]] for i in idxs]
        flat = [l for sub in lab_lists for l in sub]
        cnt = Counter(flat)
        total = len(idxs)
        for lbl, ct in cnt.items():
            rows.append({"cluster": c, "label": lbl, "freq": ct/total})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "cooccurrence.csv"), index=False)

# ─── 8. OPTIONAL: ERROR RATES BY CLUSTER ────────────────────────────────────────
def error_by_cluster(df0, labels):
    if not os.path.exists(CLASS_CSV):
        return
    dfc = pd.read_csv(CLASS_CSV)
    dfc = dfc.set_index("Unnamed: 0")
    df0_i = df0.reset_index().set_index("index")
    df_merged = df0_i.join(dfc, how="inner")
    df_merged["cluster"] = labels[df_merged.index]
    err = []
    for c in sorted(df_merged["cluster"].unique()):
        sub = df_merged[df_merged["cluster"] == c]
        rate = (sub["y_true"] != sub["y_pred"]).mean()
        err.append({"cluster": c, "error_rate": rate, "size": len(sub)})
    pd.DataFrame(err).to_csv(os.path.join(RESULTS_DIR, "error_by_cluster.csv"), index=False)

# ─── 9. VISUALIZE CLUSTERS (t‑SNE) ──────────────────────────────────────────────
def visualize_tsne(X, labels):
    X2 = X
    if X.shape[1] > 50:
        X2 = PCA(n_components=50, random_state=42).fit_transform(X)
    X2 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(X2)
    df2 = pd.DataFrame(X2, columns=("x","y"))
    df2["cluster"] = labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df2, x="x", y="y", hue="cluster", palette="tab10", s=40, alpha=0.7)
    plt.title("t‑SNE of SBERT Embeddings by Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tsne_clusters.png"))
    plt.close()

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # 1) load embeddings & labels
    X, y_true, df0, label_names = load_sbert()

    # 2) select k
    k_opt = select_k(X, k_min=2, k_max=8)

    # 3) final clustering
    labels, centers = cluster_data(X, k_opt)

    # 4) cluster–label alignment
    cluster_label_alignment(y_true, labels)

    # 5) intra vs inter variance
    variance_analysis(X, y_true, label_names)

    # 6) prototype distance histograms
    prototype_distances(X, y_true, label_names)

    # 7) co‑occurrence
    cooccurrence_by_cluster(df0, labels, y_true, label_names)

    # 8) optional error rates
    error_by_cluster(df0, labels)

    # 9) visualization
    visualize_tsne(X, labels)

    logging.info("All analyses complete. Check the results in %s", RESULTS_DIR)
=======
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import Counter
import logging
import ast
from tqdm import tqdm
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Create directory for results
os.makedirs("../results/clustering", exist_ok=True)

def load_sbert_embeddings():
    """Load SBERT embeddings and original dataset"""
    logging.info("Loading SBERT embeddings...")
    
    # Load the original dataset to get labels
    df_original = pd.read_csv("../datasets/interim/translated_output_1.csv")
    
    # Load SBERT embeddings
    try:
        sbert_data = torch.load("../datasets/interim/embeddings/pt/sbert_output_1.pt", weights_only=True)
        
        # Extract features
        features = sbert_data['features']
        if features.is_cuda:
            features = features.cpu()
        
        # Create a DataFrame with embeddings
        embedding_cols = {f'dim_{i}': features[:, i].numpy() for i in range(features.shape[1])}
        
        # Add original columns
        embedding_cols['Unnamed: 0'] = df_original['Unnamed: 0'] if 'Unnamed: 0' in df_original.columns else range(len(df_original))
        embedding_cols['CommentClass_en'] = df_original['CommentClass_en']
        embedding_cols['Comment_en'] = df_original['Comment_en']
        
        sbert_df = pd.DataFrame(embedding_cols)
        
    except Exception as e:
        logging.error(f"Error loading SBERT embeddings: {e}")
        logging.info("Trying to load from CSV...")
        sbert_df = pd.read_csv("../datasets/interim/embeddings/SBERT/sbert_embeddings.csv")
        # Add original text if available
        if 'Comment_en' not in sbert_df.columns and 'Unnamed: 0' in sbert_df.columns:
            sbert_df = sbert_df.merge(df_original[['Unnamed: 0', 'Comment_en']], on='Unnamed: 0', how='left')
    
    logging.info(f"SBERT DataFrame shape: {sbert_df.shape}")
    
    # Check for NaN values
    nan_count = sbert_df.isna().sum().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values in SBERT dataframe. Filling with zeros.")
        sbert_df = sbert_df.fillna(0)
    
    return sbert_df

def extract_embedding_features(df):
    """Extract embedding features from DataFrame"""
    feature_cols = [col for col in df.columns if col.startswith('dim_')]
    X = df[feature_cols].values
    return X

def parse_labels(df):
    """Parse the multi-label classifications"""
    labels = []
    for label in df['CommentClass_en']:
        try:
            if isinstance(label, str):
                if label.startswith('[') and label.endswith(']'):
                    parsed_label = ast.literal_eval(label)
                    if isinstance(parsed_label, list):
                        labels.append(parsed_label)
                    else:
                        labels.append([label])
                else:
                    labels.append([label])
            else:
                labels.append([str(label)])
        except (ValueError, SyntaxError):
            labels.append([str(label)])
    return labels

def find_optimal_clusters(X, max_clusters=15):
    """Find the optimal number of clusters using silhouette score and elbow method"""
    logging.info("Finding optimal number of clusters...")
    
    silhouette_scores = []
    inertia_values = []
    calinski_scores = []
    
    # Try different numbers of clusters
    for n_clusters in tqdm(range(2, max_clusters + 1)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (for elbow method)
        inertia_values.append(kmeans.inertia_)
        
        # Calculate Calinski-Harabasz Index
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        calinski_scores.append(calinski_score)
        
        logging.info(f"Clusters: {n_clusters}, Silhouette: {silhouette_avg:.4f}, Inertia: {kmeans.inertia_:.4f}, Calinski-Harabasz: {calinski_score:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    # Plot elbow method
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), inertia_values, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    
    # Plot Calinski-Harabasz Index
    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_clusters + 1), calinski_scores, marker='o')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('../results/clustering/optimal_clusters.png')
    
    # Find optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    logging.info(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
    
    return optimal_clusters

def perform_clustering(X, n_clusters):
    """Perform K-Means clustering"""
    logging.info(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels, kmeans

def analyze_clusters(df, cluster_labels, parsed_labels):
    """Analyze clusters for emotion/sentiment patterns"""
    logging.info("Analyzing clusters for emotion/sentiment patterns...")
    
    # Add cluster labels to DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Analyze label distribution within each cluster
    cluster_analysis = {}
    
    for cluster_id in range(max(cluster_labels) + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_labels_list = [parsed_labels[i] for i in cluster_indices]
        
        # Flatten the list of labels
        flat_labels = [label for sublist in cluster_labels_list for label in sublist]
        
        # Count label occurrences
        label_counts = Counter(flat_labels)
        
        # Store analysis
        cluster_analysis[cluster_id] = {
            'size': len(cluster_indices),
            'label_distribution': label_counts,
            'top_labels': label_counts.most_common(5),
            'sample_indices': cluster_indices[:10].tolist()  # Store some sample indices
        }
        
        logging.info(f"Cluster {cluster_id}: Size = {len(cluster_indices)}")
        logging.info(f"Top labels: {label_counts.most_common(5)}")
        
        # Sample comments from this cluster
        sample_comments = df.iloc[cluster_indices[:5]]['Comment_en'].tolist()
        logging.info("Sample comments:")
        for comment in sample_comments:
            logging.info(f"- {comment[:100]}...")
        logging.info("-" * 50)
    
    return cluster_analysis, df_with_clusters

def visualize_clusters(X, cluster_labels, parsed_labels, kmeans):
    """Visualize clusters using dimensionality reduction"""
    logging.info("Visualizing clusters...")
    
    # Reduce dimensionality for visualization
    # First use PCA to reduce to 50 dimensions (if more than 50)
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)
        logging.info(f"Reduced dimensions with PCA: {X.shape[1]} -> 50")
    else:
        X_reduced = X
    
    # Then use t-SNE for final 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X_reduced)
    
    # Create a DataFrame for plotting
    tsne_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'cluster': cluster_labels
    })
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', palette='viridis', s=50, alpha=0.7)
    
    # Plot cluster centers
    if X.shape[1] > 2:
        # For cluster centers, we need to include them in the original t-SNE computation
        # or use an approximation method. Here's a simple approach:
        
        # Get the indices of the points closest to each cluster center
        closest_indices = []
        for center in kmeans.cluster_centers_:
            # Calculate distances from all points to this center
            distances = np.linalg.norm(X - center, axis=1)
            # Find the index of the closest point
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        # Use the t-SNE coordinates of these closest points as approximations for centers
        centers_tsne = X_tsne[closest_indices]
        
        plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], s=200, c='red', marker='X', alpha=0.8)
    
    plt.title('t-SNE Visualization of Clusters')
    plt.savefig('../results/clustering/tsne_clusters.png')
    
    # Create a more detailed visualization with label information
    # Create a flattened list of primary labels for each point
    primary_labels = []
    for labels_list in parsed_labels:
        primary_labels.append(labels_list[0] if labels_list else "Unknown")
    
    # Add to DataFrame
    tsne_df['primary_label'] = primary_labels
    
    # Plot by primary label
    plt.figure(figsize=(14, 12))
    sns.scatterplot(data=tsne_df, x='x', y='y', hue='primary_label', palette='tab20', s=50, alpha=0.7)
    plt.title('t-SNE Visualization by Primary Label')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../results/clustering/tsne_labels.png')
    
    return tsne_df


def validate_clusters(cluster_analysis, parsed_labels):
    """Validate if clusters align with known sentiment/emotion categories"""
    logging.info("Validating clusters against known categories...")
    
    # Get all unique labels
    all_labels = set()
    for labels_list in parsed_labels:
        all_labels.update(labels_list)
    
    # Calculate purity for each cluster
    cluster_purity = {}
    label_cluster_mapping = {}
    
    for cluster_id, analysis in cluster_analysis.items():
        total_items = analysis['size']
        if total_items == 0:
            cluster_purity[cluster_id] = 0
            continue
        
        # Get the most common label in this cluster
        most_common_label, count = analysis['top_labels'][0] if analysis['top_labels'] else ("Unknown", 0)
        purity = count / total_items
        cluster_purity[cluster_id] = purity
        
        # Map labels to their most representative cluster
        for label, count in analysis['label_distribution'].items():
            if label not in label_cluster_mapping or count > label_cluster_mapping[label][1]:
                label_cluster_mapping[label] = (cluster_id, count)
    
    # Calculate overall purity
    overall_purity = sum(cluster_purity.values()) / len(cluster_purity)
    
    logging.info(f"Overall cluster purity: {overall_purity:.4f}")
    logging.info("Cluster purity by cluster:")
    for cluster_id, purity in cluster_purity.items():
        logging.info(f"Cluster {cluster_id}: {purity:.4f}")
    
    logging.info("Label to cluster mapping:")
    for label, (cluster_id, count) in label_cluster_mapping.items():
        logging.info(f"Label '{label}' -> Cluster {cluster_id} (count: {count})")
    
    # Create a heatmap of label distribution across clusters
    label_cluster_matrix = np.zeros((len(all_labels), len(cluster_analysis)))
    labels_list = sorted(all_labels)
    
    for i, label in enumerate(labels_list):
        for cluster_id, analysis in cluster_analysis.items():
            label_cluster_matrix[i, cluster_id] = analysis['label_distribution'].get(label, 0)
    
    # Normalize by cluster size
    for cluster_id, analysis in cluster_analysis.items():
        if analysis['size'] > 0:
            label_cluster_matrix[:, cluster_id] /= analysis['size']
    
    # Plot heatmap
    plt.figure(figsize=(12, max(8, len(all_labels) * 0.4)))
    sns.heatmap(label_cluster_matrix, xticklabels=[f"Cluster {i}" for i in range(len(cluster_analysis))],
                yticklabels=labels_list, cmap="YlGnBu", annot=False)
    plt.title('Label Distribution Across Clusters (Normalized)')
    plt.tight_layout()
    plt.savefig('../results/clustering/label_cluster_heatmap.png')
    
    return {
        'cluster_purity': cluster_purity,
        'overall_purity': overall_purity,
        'label_cluster_mapping': label_cluster_mapping
    }

def emotion_aware_summarization(df_with_clusters, cluster_analysis):
    """Generate emotion-aware summaries for each cluster"""
    logging.info("Generating emotion-aware summaries for each cluster...")
    
    summaries = {}
    
    for cluster_id, analysis in cluster_analysis.items():
        # Get cluster data
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Get top emotions/sentiments
        top_emotions = [label for label, _ in analysis['top_labels']]
        
        # Calculate average review length
        avg_length = cluster_data['Comment_en'].str.len().mean()
        
        # Get sample reviews (prioritize shorter ones for readability)
        sample_reviews = (
            cluster_data
            .sort_values(by='Comment_en', key=lambda x: x.str.len())
            ['Comment_en']
            .head(5)
            .tolist()
        )
        
        # Create summary
        summary = {
            'cluster_id': cluster_id,
            'size': analysis['size'],
            'top_emotions': top_emotions,
            'avg_review_length': avg_length,
            'sample_reviews': sample_reviews,
            'emotion_profile': dict(analysis['top_labels'])
        }
        
        summaries[cluster_id] = summary
        
        logging.info(f"Cluster {cluster_id} Summary:")
        logging.info(f"Size: {analysis['size']} reviews")
        logging.info(f"Top emotions: {', '.join(top_emotions)}")
        logging.info(f"Average review length: {avg_length:.1f} characters")
        logging.info("Sample reviews:")
        for i, review in enumerate(sample_reviews[:3], 1):
            logging.info(f"{i}. {review[:100]}...")
        logging.info("-" * 50)
    
    # Save summaries to CSV
    summary_rows = []
    for cluster_id, summary in summaries.items():
        row = {
            'Cluster': cluster_id,
            'Size': summary['size'],
            'Top Emotions': ', '.join(summary['top_emotions']),
            'Avg Review Length': summary['avg_review_length'],
            'Sample Review': summary['sample_reviews'][0] if summary['sample_reviews'] else ''
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('../results/clustering/cluster_summaries.csv', index=False)
    logging.info("Saved cluster summaries to '../results/clustering/cluster_summaries.csv'")
    
    return summaries

def recommend_products_by_emotion(df_with_clusters, cluster_analysis):
    """Generate emotion-based product recommendations"""
    logging.info("Generating emotion-based product recommendations...")
    
    # This is a simplified implementation that assumes:
    # 1. We have product information in the dataset
    # 2. We can map emotions to product categories
    
    # In a real implementation, you would:
    # 1. Extract product information from the reviews
    # 2. Analyze which products are mentioned positively in each emotion cluster
    # 3. Create a recommendation system based on emotion-product affinities
    
    # For demonstration, we'll create a simple mapping
    emotion_product_affinities = {
        'positive': ['premium products', 'gift items', 'luxury items'],
        'negative': ['budget alternatives', 'customer service contact', 'warranty information'],
        'neutral': ['bestsellers', 'most reviewed items'],
        'excited': ['new arrivals', 'trending items', 'limited editions'],
        'disappointed': ['discounted items', 'alternatives', 'customer support'],
        'satisfied': ['similar products', 'complementary items'],
        'angry': ['customer service', 'refund policy', 'alternatives']
    }
    
    # Map clusters to recommended products based on dominant emotions
    recommendations = {}
    
    for cluster_id, analysis in cluster_analysis.items():
        top_emotions = [label.lower() for label, _ in analysis['top_labels']]
        
        # Find matching emotions in our affinity map
        matching_products = []
        for emotion in top_emotions:
            for key in emotion_product_affinities:
                if key in emotion or emotion in key:
                    matching_products.extend(emotion_product_affinities[key])
        
        # Remove duplicates
        matching_products = list(set(matching_products))
        
        recommendations[cluster_id] = {
            'dominant_emotions': top_emotions,
            'recommended_product_categories': matching_products
        }
        
        logging.info(f"Cluster {cluster_id} Recommendations:")
        logging.info(f"Dominant emotions: {', '.join(top_emotions)}")
        logging.info(f"Recommended product categories: {', '.join(matching_products)}")
        logging.info("-" * 50)
    
    # Save recommendations to CSV
    recommendation_rows = []
    for cluster_id, rec in recommendations.items():
        row = {
            'Cluster': cluster_id,
            'Dominant Emotions': ', '.join(rec['dominant_emotions']),
            'Recommended Product Categories': ', '.join(rec['recommended_product_categories'])
        }
        recommendation_rows.append(row)
    
    recommendation_df = pd.DataFrame(recommendation_rows)
    recommendation_df.to_csv('../results/clustering/emotion_recommendations.csv', index=False)
    logging.info("Saved emotion-based recommendations to '../results/clustering/emotion_recommendations.csv'")
    
    return recommendations

def main():
    """Main function to run the clustering and emotion analysis"""
    logging.info("Starting clustering and emotion analysis...")
    
    # Load SBERT embeddings
    sbert_df = load_sbert_embeddings()
    
    # Extract features and parse labels
    X = extract_embedding_features(sbert_df)
    parsed_labels = parse_labels(sbert_df)
    
    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(X)
    
    # Perform clustering
    cluster_labels, kmeans_model = perform_clustering(X, optimal_clusters)
    
    # Analyze clusters
    cluster_analysis, df_with_clusters = analyze_clusters(sbert_df, cluster_labels, parsed_labels)
    
    # Visualize clusters
    tsne_df = visualize_clusters(X, cluster_labels, parsed_labels, kmeans_model)
    
    # Validate clusters
    validation_results = validate_clusters(cluster_analysis, parsed_labels)
    
    # Generate emotion-aware summaries
    summaries = emotion_aware_summarization(df_with_clusters, cluster_analysis)
    
    # Generate emotion-based recommendations
    recommendations = recommend_products_by_emotion(df_with_clusters, cluster_analysis)
    
    # Save the clustered data
    df_with_clusters.to_csv('../results/clustering/clustered_data.csv', index=False)
    logging.info("Saved clustered data to '../results/clustering/clustered_data.csv'")
    
    # Save t-SNE coordinates
    tsne_df.to_csv('../results/clustering/tsne_coordinates.csv', index=False)
    logging.info("Saved t-SNE coordinates to '../results/clustering/tsne_coordinates.csv'")
    
    logging.info("Clustering and emotion analysis completed!")
    
    return {
        'cluster_analysis': cluster_analysis,
        'validation_results': validation_results,
        'summaries': summaries,
        'recommendations': recommendations
    }
>>>>>>> c9911334d70faef7ab964419f234f9311fcf8d1d

if __name__ == "__main__":
    main()
