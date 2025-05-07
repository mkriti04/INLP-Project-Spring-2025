import os
import ast
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# === 1. Helper Functions ===
def safe_parse(lst_str):
    """Safely parse string representations of lists"""
    try:
        if isinstance(lst_str, list):
            return lst_str
        return ast.literal_eval(lst_str)
    except:
        return [lst_str]

# === 2. Load SBERT Embeddings ===
def load_embeddings():
    """Load document embeddings from either PT or CSV files"""
    logging.info("Loading SBERT embeddings...")
    
    try:
        # First try to load from PT file
        data = torch.load("../datasets/interim/pt/sbert_output_1.pt")
        embeddings = data['embeddings'].numpy()
        labels = data['labels']
        indices = data['indices']
        
        # Create DataFrame
        df = pd.DataFrame(embeddings)
        df.insert(0, 'original_index', indices)
        df['labels'] = labels
        logging.info(f"Successfully loaded embeddings from PT file: {embeddings.shape}")
        
    except Exception as e:
        # Fallback to CSV file
        logging.info(f"Failed to load from PT file: {e}")

    # Ensure labels are in the correct format
    df['labels'] = df['labels'].apply(safe_parse)
    return df

# === 3. Neural Network Models ===

class SimpleNN(nn.Module):
    """Simple Feed-Forward Neural Network for multi-label classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)


class DeepNN(nn.Module):
    """Deep Neural Network with multiple hidden layers"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.4):
        super(DeepNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Add output layer with sigmoid for multi-label
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


# === 4. Training and Evaluation ===

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training and evaluation"""
    # Extract features and labels
    feature_cols = df.columns[1:-1]  # All columns except index and labels
    X = df[feature_cols].values
    
    # Process labels
    y_raw = df['labels'].tolist()
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logging.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Classes: {mlb.classes_}")
    
    return X_train, X_test, y_train, y_test, mlb.classes_


def train_model(model, X_train, y_train, X_test, y_test, 
                batch_size=32, lr=0.001, num_epochs=20, patience=3):
    """Train and evaluate a PyTorch model"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "../models/best_sbert_classifier.pt")
            else:
                patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Train Loss: {avg_train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}')
            
        # Check if early stopping criteria is met
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("../models/best_sbert_classifier.pt"))
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_binary = (y_pred > 0.5).float().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # Get per-class metrics
    class_report = classification_report(y_test, y_pred_binary, 
                                         zero_division=0, output_dict=True)
    
    return model, metrics, train_losses, val_losses, class_report


def plot_training_curves(train_losses, val_losses, model_name):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../results/sbert_{model_name}_loss_curve.png")
    plt.close()


def plot_metrics_comparison(models_metrics, model_names):
    """Plot performance metrics comparison between models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data for plotting
    data = []
    for i, metrics_dict in enumerate(models_metrics):
        for metric in metrics:
            data.append({
                'Model': model_names[i],
                'Metric': metric.capitalize(),
                'Value': metrics_dict[metric]
            })
    
    df_plot = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_plot)
    plt.title('Performance Metrics Comparison')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("../results/sbert_metrics_comparison.png")
    plt.close()


# === 5. Main Function ===

def main():
    """Main function to run the classification pipeline"""
    logging.info("Starting SBERT embeddings classification")
    
    # Load embeddings
    df = load_embeddings()
    logging.info(f"Loaded dataframe with shape: {df.shape}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, classes = prepare_data(df)
    
    # Model parameters
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Define models to test
    models = [
        {
            'name': 'SimpleNN',
            'model': SimpleNN(input_dim, hidden_dim=128, output_dim=output_dim)
        },
        {
            'name': 'DeepNN',
            'model': DeepNN(
                input_dim, 
                hidden_dims=[256, 128, 64], 
                output_dim=output_dim
            )
        }
    ]
    
    # Train and evaluate each model
    results = []
    model_names = []
    
    for model_info in models:
        name = model_info['name']
        model = model_info['model']
        
        logging.info(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        
        trained_model, metrics, train_losses, val_losses, class_report = train_model(
            model, X_train, y_train, X_test, y_test
        )
        
        # Log results
        logging.info(f"Results for {name}:")
        for metric, value in metrics.items():
            logging.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot learning curves
        plot_training_curves(train_losses, val_losses, name)
        
        # Save detailed class report
        pd.DataFrame(class_report).transpose().to_csv(
            f"../results/sbert_{name}_class_report.csv"
        )
        
        # Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_names': classes,
            'metrics': metrics,
            'architecture': str(trained_model)
        }, f"../models/{name}_sbert_classifier.pt")
        
        results.append(metrics)
        model_names.append(name)
    
    # Compare models
    plot_metrics_comparison(results, model_names)
    
    # Save overall results
    pd.DataFrame({
        model_names[i]: result for i, result in enumerate(results)
    }).to_csv("../results/sbert_model_comparison.csv")
    
    logging.info("Classification completed! Results saved to 'results' directory.")


if __name__ == "__main__":
    main()