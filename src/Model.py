import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ast
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Create directory for results
os.makedirs("../results", exist_ok=True)

# Load the original dataset to get labels
df_original = pd.read_csv("../datasets/interim/translated_output_1.csv")

# Load embeddings from .pt files
# Modified load_embeddings_from_pt function
def load_embeddings_from_pt(file_path, df_original):
    data = torch.load(file_path, weights_only=True)  # Add weights_only=True for security
    word_vectors = {}
    
    # Extract word vectors from the loaded data
    features = data['features']
    labels = data['labels']  # These should be the vocabulary words
    
    # Create a dictionary mapping words to their vectors
    for i, word in enumerate(labels):
        word_vectors[word] = features[i]
    
    # Create document embeddings by averaging word vectors
    # First, tokenize the comments from the original dataframe
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    
    # Download necessary NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        if isinstance(text, str):
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
            return tokens
        return []
    
    # Tokenize comments
    df_original['tokenized_comments'] = df_original['Comment_en'].fillna("").apply(preprocess_text)
    
    # Function to create document vector by averaging word vectors
    def document_vector(tokens):
        doc_vectors = [word_vectors[word] for word in tokens if word in word_vectors]
        if doc_vectors:
            return torch.mean(torch.stack(doc_vectors), dim=0)
        else:
            # Return zero vector if no words from the document are in the vocabulary
            return torch.zeros(features.shape[1])
    
    # Create document embeddings
    document_embeddings = df_original['tokenized_comments'].apply(document_vector)
    
    # Create a dataframe with document embeddings
    embeddings_df = pd.DataFrame()
    
    # Add the original ID and class labels
    embeddings_df['Unnamed: 0'] = df_original['Unnamed: 0'] if 'Unnamed: 0' in df_original.columns else range(len(df_original))
    embeddings_df['CommentClass_en'] = df_original['CommentClass_en']
    
    # Convert document embeddings to numpy arrays and add as columns
    document_embeddings_array = torch.stack(document_embeddings.tolist()).numpy()
    for i in range(document_embeddings_array.shape[1]):
        embeddings_df[f'dim_{i}'] = document_embeddings_array[:, i]
    
    return embeddings_df


# Load embeddings
logging.info("Loading embeddings...")
    # Try to load from .pt files first
cbow_df = load_embeddings_from_pt("../datasets/interim/embeddings/pt/cbow_output_1.pt", df_original)
skipgram_df = load_embeddings_from_pt("../datasets/interim/embeddings/pt/skipgram_output_1.pt", df_original)
tfidf_df = load_embeddings_from_pt("../datasets/interim/embeddings/pt/tfidf_output_1.pt", df_original)
sbert_df = load_embeddings_from_pt("../datasets/interim/embeddings/pt/sbert_output_1.pt", df_original)

    
# Load TF-IDF from CSV (assuming it's already processed)
# tfidf_df = pd.read_csv("../datasets/tfidf_embeddings_2.csv")
    
# # For SBERT, check if it exists as .pt or try to load from CSV
# try:
#     sbert_df = load_embeddings_from_pt("../models/sbert_contrastive.pt", df_original)
# except:
#     sbert_df = pd.read_csv("../datasets/exterim/embeddings/SBERT/sbert_embeddings.csv")
    
# except Exception as e:
#     logging.error(f"Error loading embeddings: {e}")
#     logging.info("Falling back to CSV files...")
    
#     # Fallback to CSV files
#     tfidf_df = pd.read_csv("../datasets/tfidf_embeddings_2.csv")
#     cbow_df = pd.read_csv("../datasets/exterim/embeddings/Cbow/cbow_embeddings.csv")
#     skipgram_df = pd.read_csv("../datasets/exterim/embeddings/Skipgram/skipgram_embeddings.csv")
#     sbert_df = pd.read_csv("../datasets/exterim/embeddings/SBERT/sbert_embeddings.csv")

# Function to prepare data for classification
def prepare_data(df, label_col='CommentClass_en', id_col='Unnamed: 0'):
    # Extract features (all columns except label and ID)
    feature_cols = [col for col in df.columns if col not in [label_col, id_col]]
    X = df[feature_cols].values
    
    # Handle multi-label classification
    # If labels are stored as strings (like "[1, 2, 3]"), convert to actual lists
    if df[label_col].dtype == 'object' and isinstance(df[label_col].iloc[0], str) and df[label_col].iloc[0].startswith('['):
        y_raw = df[label_col].apply(ast.literal_eval).tolist()
    else:
        # If single label, convert to list format for consistency
        y_raw = df[label_col].apply(lambda x: [x]).tolist()
    
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    
    return X, y, mlb.classes_

# Neural Network Model for Multi-Label Classification
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer3(x))
        return x

# Function to train and evaluate neural network
def train_evaluate_nn(X_train, y_train, X_test, y_test, input_dim, output_dim, embedding_name):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    hidden_dim = min(512, max(64, input_dim // 2))  # Adaptive hidden dimension
    model = MultiLabelClassifier(input_dim, hidden_dim, output_dim)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_binary = (y_pred > 0.5).float().numpy()
        
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    
    logging.info(f"Neural Network Results for {embedding_name}:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model
    }

# Function to train and evaluate traditional ML models
def train_evaluate_ml(X_train, y_train, X_test, y_test, embedding_name, model_type='svm'):
    if model_type == 'svm':
        model = OneVsRestClassifier(LinearSVC(random_state=42))
    elif model_type == 'rf':
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logging.info(f"{model_type.upper()} Results for {embedding_name}:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model
    }

# Main comparison function
def compare_embeddings():
    # Dictionary to store results
    results = defaultdict(dict)
    
    # List of embedding dataframes and their names
    embedding_dfs = [
        (tfidf_df, "TF-IDF"),
        (cbow_df, "CBOW"),
        (skipgram_df, "Skip-gram"),
        (sbert_df, "SBERT")
    ]
    
    # For each embedding type
    for df, name in embedding_dfs:
        logging.info(f"\n{'='*50}\nProcessing {name} embeddings\n{'='*50}")
        
        # Prepare data
        X, y, classes = prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info(f"Data shape: X={X.shape}, y={y.shape}")
        logging.info(f"Classes: {classes}")
        
        # Train and evaluate models
        results[name]['nn'] = train_evaluate_nn(
            X_train, y_train, X_test, y_test, X.shape[1], y.shape[1], name
        )
        
        results[name]['svm'] = train_evaluate_ml(
            X_train, y_train, X_test, y_test, name, model_type='svm'
        )
        
        results[name]['rf'] = train_evaluate_ml(
            X_train, y_train, X_test, y_test, name, model_type='rf'
        )
    
    return results

# Visualize results
def visualize_results(results):
    # Prepare data for plotting
    embedding_names = list(results.keys())
    model_types = ['nn', 'svm', 'rf']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Data for this metric
        data = []
        for embedding in embedding_names:
            for model in model_types:
                data.append({
                    'Embedding': embedding,
                    'Model': model.upper(),
                    metric.capitalize(): results[embedding][model][metric]
                })
        
        # Create DataFrame
        df_plot = pd.DataFrame(data)
        
        # Plot
        sns.barplot(x='Embedding', y=metric.capitalize(), hue='Model', data=df_plot, ax=axes[i])
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig('../results/embedding_comparison.png')
    logging.info("Results visualization saved to '../results/embedding_comparison.png'")
    
    # Save detailed results to CSV
    detailed_results = []
    for embedding in embedding_names:
        for model in model_types:
            row = {
                'Embedding': embedding,
                'Model': model.upper()
            }
            for metric in metrics:
                row[metric.capitalize()] = results[embedding][model][metric]
            detailed_results.append(row)
    
    pd.DataFrame(detailed_results).to_csv('../results/embedding_comparison_results.csv', index=False)
    logging.info("Detailed results saved to '../results/embedding_comparison_results.csv'")

if __name__ == "__main__":
    logging.info("Starting embedding comparison...")
    results = compare_embeddings()
    visualize_results(results)
    logging.info("Embedding comparison completed!")
