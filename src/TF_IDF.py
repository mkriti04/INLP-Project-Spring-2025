'''
Removing stop words, converting into lower and TIF-IDF embeddings
And saving the model as PT file
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

# Create directories if they don't exist
os.makedirs("../datasets/interim/embeddings/csv", exist_ok=True)
os.makedirs("../datasets/interim/embeddings/pt", exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
print("→ Loading dataset...")
df = pd.read_csv("../datasets/interim/translated_output_2_clean.csv")
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
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Preserve original columns if they exist
if 'Unnamed: 0' in df.columns:
    tfidf_df.insert(0, 'Unnamed: 0', df['Unnamed: 0'])
else:
    tfidf_df.insert(0, 'original_index', df.index)

# Add labels column
if 'CommentClass_en' in df.columns:
    tfidf_df['CommentClass_en'] = df['CommentClass_en']

# Save as CSV (following your original approach)
csv_path = "../datasets/interim/embeddings/csv/tfidf_2.csv"
tfidf_df.to_csv(csv_path, index=False)
print(f"✓ Saved TF-IDF embeddings to CSV: {csv_path}")

# Save the TF-IDF model as PT file
# Create a dictionary with all necessary components to rebuild and use the model
model_dict = {
    'vocabulary': tfidf_vectorizer.vocabulary_,
    'idf': tfidf_vectorizer.idf_,
    'stop_words': list(tfidf_vectorizer.stop_words_) if hasattr(tfidf_vectorizer, 'stop_words_') else None,
    'feature_names': tfidf_vectorizer.get_feature_names_out(),
    # Add any other important attributes from the vectorizer
}

# Save the model dictionary as a PT file
model_path = "../datasets/interim/embeddings/pt/tfidf_model_2.pt"
torch.save(model_dict, model_path)
print(f"✓ Saved TF-IDF model as PT file: {model_path}")

# Also save the vectorizer as pickle (for completeness and ease of use)
# vectorizer_path = "../datasets/interim/embeddings/pt/tfidf_vectorizer_1.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"✓ Saved TF-IDF vectorizer as pickle: {vectorizer_path}")

# Save TF-IDF features (embeddings) in PyTorch format
feature_names = tfidf_vectorizer.get_feature_names_out()

# Extract labels if available
if 'CommentClass_en' in df.columns:
    labels = df['CommentClass_en'].tolist()
else:
    labels = None

# Save embeddings as pytorch tensor
tfidf_tensor = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
embeddings_path = "../datasets/interim/embeddings/pt/tfidf_features_2.pt"
torch.save({
    'embeddings': tfidf_tensor,
    'feature_names': feature_names,
    'labels': labels,
    'indices': df.index.tolist()
}, embeddings_path)
print(f"✓ Saved TF-IDF embeddings (.pt) with shape {tfidf_tensor.shape}")

print("\nTF-IDF processing completed successfully!")