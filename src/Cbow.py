import os
import glob
import ast
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import torch

# === 1. Setup directories ===
os.makedirs("models", exist_ok=True)
os.makedirs("datasets/interim/embeddings/csv/amazonreviews", exist_ok=True)
os.makedirs("datasets/interim/embeddings/pt/amazonreviews", exist_ok=True)

# === 2. Download NLTK data (first run only) ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === 3. Text preprocessing tools ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and tok.isalnum() and len(tok) > 2
    ]
    return tokens

def safe_parse(lst_str):
    """Safely parse string representations of lists"""
    try:
        return ast.literal_eval(lst_str)
    except:
        if isinstance(lst_str, list):
            return lst_str
        return [lst_str]

# === 4. Load and process the dataset ===
print("→ Loading dataset...")
# Replace with your actual dataset path
df = pd.read_csv("../datasets/interim/converted_amazonReviews_50k_clean.csv")

# Clean up any unnamed columns
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

# Process labels
df['labels'] = df['CommentClass_en'].apply(safe_parse)

# Process tokens
if 'tokens' in df.columns:
    df['tokens'] = df['tokens'].apply(safe_parse)
else:
    # If tokens column doesn't exist, use the clean_comment
    df['tokens'] = df['clean_comment'].fillna("").apply(preprocess)

print(f"→ Loaded dataset with {len(df)} rows")

# === 5. Collect tokens and labels
docs = df['tokens'].tolist()
lbls = df['labels'].tolist()
idxs = df.index.tolist()

# === 6. Train CBOW Word2Vec
print("→ Training Word2Vec CBOW...")
w2v = Word2Vec(
    sentences=docs,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0,  # CBOW model
    workers=4,
    epochs=10
)

# === 7. Save word embeddings (.pt)
wv = w2v.wv
word_to_index = wv.key_to_index
index_to_word = [None] * len(word_to_index)
for w, i in word_to_index.items():
    index_to_word[i] = w

emb_tensor = torch.tensor(wv.vectors)
torch.save({
    'features': emb_tensor,
    'word_to_index': word_to_index,
    'index_to_word': index_to_word
}, "../datasets/interim/embeddings/pt/amazonreviews/word_embeddings.pt")
print(f"✓ Saved word embeddings (.pt) with shape {emb_tensor.shape}")

# === 8. Compute document embeddings
def document_vector(tokens):
    """Create document embedding by averaging word vectors"""
    vecs = [wv[t] for t in tokens if t in wv]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(w2v.vector_size, dtype=float)

print("→ Building document embeddings...")
doc_embs = [document_vector(d) for d in docs]

# === 9. Save document embeddings to torch (.pt) file
doc_embs_tensor = torch.tensor(doc_embs)
labels_list = lbls
torch.save({
    'embeddings': doc_embs_tensor,
    'labels': labels_list,
    'indices': idxs
}, "../datasets/interim/embeddings/pt/amazonreviews/cbow_amazon.pt")
print(f"✓ Saved document embeddings (.pt) with shape {doc_embs_tensor.shape}")

# === 10. Save document embeddings (.csv)
# Create DataFrame with embeddings
out_df = pd.DataFrame(doc_embs)
# Add original indices and labels
out_df.insert(0, 'original_index', idxs)
out_df['labels'] = [str(l) for l in lbls]  # Convert labels to string for CSV storage
out_path = "../datasets/interim/embeddings/csv/amazonreviews/cbow_amazon.csv"
out_df.to_csv(out_path, index=False)
print(f"✓ Saved document embeddings (.csv) → {out_path}")