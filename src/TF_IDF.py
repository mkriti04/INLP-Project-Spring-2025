'''
Removing stop words, converting into lower and TIF-IDF embeddings
And saving the model as PT file - Memory-optimized version
'''
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import pickle
import numpy as np

# Create directories if they don't exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../datasets/interim/embeddings/csv/amazonreviews", exist_ok=True)
os.makedirs("../datasets/interim/embeddings/pt/amazonreviews", exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
print("→ Loading dataset...")
df = pd.read_csv("../datasets/interim/converted_amazonReviews_50k_clean.csv")
print(f"Loaded dataset with {len(df)} rows")

# Print column names to debug
print("Available columns:", df.columns.tolist())

# Initialize NLTK components
stop_words = set(stopwords.words('english'))  # Load English stopwords
lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer

def custom_tokenizer(text):
    text = str(text).lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens  # Returns lemmatized tokens

# Check if 'Comment_en' column exists, otherwise look for alternatives
if 'Comment_en' in df.columns:
    text_column = 'Comment_en'
else:
    # Try to find a suitable text column
    text_columns = [col for col in df.columns if 'comment' in col.lower() or 'text' in col.lower()]
    if text_columns:
        text_column = text_columns[0]
    else:
        raise ValueError("Could not find a suitable text column for TF-IDF")

print(f"Using '{text_column}' for TF-IDF processing")

# Add special tokens to the text (as in your original code)
df[text_column] = df[text_column].fillna("<UNK>").apply(lambda x: f"<s> {x} </s>")

# Display a sample of the preprocessed text
print("\nSample of preprocessed text:")
for i in range(min(3, len(df))):
    print(f"{df[text_column].iloc[i][:100]}...")

# Initialize TF-IDF Vectorizer with custom tokenizer
print("\n→ Creating TF-IDF vectors...")
# Use float32 instead of float64 to reduce memory usage
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None, dtype=np.float32)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Get feature names for later use
feature_names = tfidf_vectorizer.get_feature_names_out()

# === APPROACH 1: Process in batches for CSV ===
# Instead of converting the entire matrix at once, save in chunks
print("\n→ Saving TF-IDF embeddings to CSV in batches...")
batch_size = 1000  # Adjust based on your available memory
csv_path = "../datasets/interim/embeddings/csv/amazonreviews/tfidf_amazonreview.csv"

# Create an index frame first (with original index if available)
if 'Unnamed: 0' in df.columns:
    index_df = pd.DataFrame({'Unnamed: 0': df['Unnamed: 0']})
else:
    index_df = pd.DataFrame({'original_index': df.index})

# Add labels if available
if 'CommentClass_en' in df.columns:
    index_df['CommentClass_en'] = df['CommentClass_en']

# Write the header row with column names
header = list(index_df.columns) + list(feature_names)
with open(csv_path, 'w') as f:
    f.write(','.join(header) + '\n')

# Process in batches
for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    print(f"Processing batch {batch_start//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
    
    # Extract batch of sparse matrix and convert to dense
    batch_matrix = tfidf_matrix[batch_start:batch_end].toarray()
    
    # Create batch dataframe with index columns
    batch_df = index_df.iloc[batch_start:batch_end].copy()
    
    # Create DataFrame from the TF-IDF matrix batch directly
    tfidf_batch_df = pd.DataFrame(
        batch_matrix, 
        columns=feature_names,
        index=batch_df.index
    )
    
    # Concatenate the index dataframe and TF-IDF dataframe
    batch_df = pd.concat([batch_df.reset_index(drop=True), 
                         tfidf_batch_df.reset_index(drop=True)], axis=1)
    
    # Append to CSV (without header after first batch)
    batch_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Clear memory
    del batch_matrix
    del batch_df

print(f"✓ Saved TF-IDF embeddings to CSV: {csv_path}")

# === APPROACH 2: Save model components separately ===
# Save the TF-IDF model as PT file
model_dict = {
    'vocabulary': tfidf_vectorizer.vocabulary_,
    'idf': tfidf_vectorizer.idf_,
    'stop_words': list(tfidf_vectorizer.stop_words_) if hasattr(tfidf_vectorizer, 'stop_words_') else None,
    'feature_names': feature_names,
}

# Save the model dictionary as a PT file
model_path = "../datasets/interim/embeddings/pt/amazonreviews/tfidf_model.pt"
torch.save(model_dict, model_path)
print(f"✓ Saved TF-IDF model as PT file: {model_path}")

# Also save the vectorizer as pickle (for completeness and ease of use)
vectorizer_path = "../models/tfidf_vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"✓ Saved TF-IDF vectorizer as pickle: {vectorizer_path}")

# === APPROACH 3: Save as sparse tensor in PyTorch format ===
# Extract labels if available
if 'CommentClass_en' in df.columns:
    labels = df['CommentClass_en'].tolist()
else:
    labels = None

# Convert sparse matrix to sparse tensor (saving memory)
print("\n→ Converting to sparse PyTorch tensor...")
coo = tfidf_matrix.tocoo()
indices = torch.LongTensor([coo.row, coo.col])
values = torch.FloatTensor(coo.data)
shape = coo.shape

# Create sparse tensor
sparse_tfidf_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

# Save sparse embeddings tensor
embeddings_path = "../datasets/interim/embeddings/pt/amazonreviews/tfidf_amazonreview_sparse.pt"
torch.save({
    'embeddings': sparse_tfidf_tensor,
    'feature_names': feature_names,
    'labels': labels,
    'indices': df.index.tolist()
}, embeddings_path)
print(f"✓ Saved sparse TF-IDF embeddings (.pt) with shape {sparse_tfidf_tensor.shape}")

print("\nTF-IDF processing completed successfully!")