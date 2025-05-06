import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Create directory for results
os.makedirs("../results", exist_ok=True)

# Improved Neural Network for Multi-Label Classification
class ImprovedMultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(ImprovedMultiLabelClassifier, self).__init__()
        
        # Create a list of layers with specified dimensions
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply the model and sigmoid activation for multi-label
        return torch.sigmoid(self.model(x))

def prepare_data(df, label_col='CommentClass_en', id_col='Unnamed: 0'):
    # Extract features (all columns except label and ID)
    feature_cols = [col for col in df.columns if col not in [label_col, id_col]]
    X = df[feature_cols].values
    
    # Handle multi-label classification
    y_raw = []
    for label in df[label_col]:
        try:
            if isinstance(label, str):
                if label.startswith('[') and label.endswith(']'):
                    # Try to parse as a list
                    parsed_label = ast.literal_eval(label)
                    if isinstance(parsed_label, list):
                        y_raw.append(parsed_label)
                    else:
                        y_raw.append([label])
                else:
                    y_raw.append([label])
            else:
                y_raw.append([str(label)])
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a single label
            y_raw.append([str(label)])
    
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    
    # Print some debugging information
    logging.info(f"Feature shape: {X.shape}, Label shape: {y.shape}")
    logging.info(f"Label classes: {mlb.classes_}")
    logging.info(f"Sample labels: {y_raw[:5]}")
    
    return X, y, mlb.classes_

def train_improved_model(X_train, y_train, X_val, y_val, input_dim, output_dim, class_names):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Check for NaN values
    if torch.isnan(X_train_tensor).any():
        logging.warning("NaN values found in training features. Replacing with zeros.")
        X_train_tensor = torch.nan_to_num(X_train_tensor)
    
    if torch.isnan(X_val_tensor).any():
        logging.warning("NaN values found in validation features. Replacing with zeros.")
        X_val_tensor = torch.nan_to_num(X_val_tensor)
    
    # Handle class imbalance with weighted sampling
    class_counts = y_train.sum(axis=0)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.sum(y_train * class_weights, axis=1)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model with multiple hidden layers
    hidden_dims = [512, 256, 128]
    model = ImprovedMultiLabelClassifier(input_dim, hidden_dims, output_dim)
    
    # Loss and optimizer
    # Use BCEWithLogitsLoss if you remove sigmoid from the model's forward method
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop with early stopping
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # For plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), '../results/best_sbert_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('../results/sbert_training_loss.png')
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('../results/best_sbert_model.pt'))
    
    return model

def evaluate_model(model, X_test, y_test, class_names):
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Replace NaN values
    if torch.isnan(X_test_tensor).any():
        X_test_tensor = torch.nan_to_num(X_test_tensor)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor)
        y_pred_binary = (y_pred_probs > 0.5).float().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    per_class_precision = precision_score(y_test, y_pred_binary, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred_binary, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred_binary, average=None, zero_division=0)
    
    # Print overall metrics
    logging.info("Overall Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    
    # Print per-class metrics
    logging.info("\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        logging.info(f"Class: {class_name}")
        logging.info(f"  Precision: {per_class_precision[i]:.4f}")
        logging.info(f"  Recall: {per_class_recall[i]:.4f}")
        logging.info(f"  F1 Score: {per_class_f1[i]:.4f}")
    
    # Create a confusion matrix for each class
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))
        cm = np.zeros((2, 2))
        cm[0, 0] = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 0))  # TN
        cm[0, 1] = np.sum((y_test[:, i] == 0) & (y_pred_binary[:, i] == 1))  # FP
        cm[1, 0] = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 0))  # FN
        cm[1, 1] = np.sum((y_test[:, i] == 1) & (y_pred_binary[:, i] == 1))  # TP
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {class_name}')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(f'../results/confusion_matrix_{class_name.replace("/", "_")}.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
def main():
    logging.info("Loading SBERT embeddings...")
    
    # Load the original dataset to get labels
    df_original = pd.read_csv("../datasets/interim/translated_output_1.csv")
    
    # Load SBERT embeddings
    try:
        sbert_data = torch.load("../datasets/interim/embeddings/pt/sbert_output_1.pt", weights_only=True)
        
        # Extract features and labels
        features = sbert_data['features']
        if features.is_cuda:
            features = features.cpu()
        
        # Create a DataFrame with embeddings
        embedding_cols = {f'dim_{i}': features[:, i].numpy() for i in range(features.shape[1])}
        
        # Add original columns
        embedding_cols['Unnamed: 0'] = df_original['Unnamed: 0'] if 'Unnamed: 0' in df_original.columns else range(len(df_original))
        embedding_cols['CommentClass_en'] = df_original['CommentClass_en']
        
        sbert_df = pd.DataFrame(embedding_cols)
        
    except Exception as e:
        logging.error(f"Error loading SBERT embeddings: {e}")
        logging.info("Trying to load from CSV...")
        sbert_df = pd.read_csv("../datasets/interim/embeddings/SBERT/sbert_embeddings.csv")
    
    logging.info(f"SBERT DataFrame shape: {sbert_df.shape}")
    
    # Check for NaN values
    nan_count = sbert_df.isna().sum().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values in SBERT dataframe. Filling with zeros.")
        sbert_df = sbert_df.fillna(0)
    
    # Prepare data
    X, y, class_names = prepare_data(sbert_df)
    
    # Check class distribution
    class_counts = np.sum(y, axis=0)
    logging.info(f"Class distribution: {class_counts}")
    
    # Split without stratification - completely avoid the stratification issue
    logging.info("Splitting data without stratification due to rare classes...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train the model
    logging.info("Training the model...")
    model = train_improved_model(
        X_train, y_train, 
        X_val, y_val, 
        X_train.shape[1], y_train.shape[1], 
        class_names
    )
    
    # Evaluate the model
    logging.info("Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('../results/sbert_metrics.csv', index=False)
    logging.info(f"Metrics saved to '../results/sbert_metrics.csv'")
    
    logging.info("SBERT multi-label classification completed!")

if __name__ == "__main__":
    main()
