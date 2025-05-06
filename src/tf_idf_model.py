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

# === 2. Load Embeddings ===
def load_embeddings(embedding_type):
    """Load document embeddings from either PT or CSV files
    
    Args:
        embedding_type: 'cbow', 'skipgram', or 'tfidf'
    """
    logging.info(f"Loading {embedding_type.upper()} embeddings...")
    
    try:
        # First try to load from PT file
        if embedding_type == 'tfidf':
            file_name = "tfidf_features_2.pt"
        else:
            file_name = f"{embedding_type}_model_2.pt"
            
        data = torch.load(f"../datasets/interim/embeddings/pt/{file_name}")
        embeddings = data['embeddings'].numpy()
        labels = data['labels']
        indices = data['indices']
        
        # Create DataFrame
        df = pd.DataFrame(embeddings)
        df.insert(0, 'original_index', indices)
        df['labels'] = labels
        logging.info(f"Successfully loaded {embedding_type} embeddings from PT file: {embeddings.shape}")
        
    except Exception as e:
        # Fallback to CSV file
        logging.info(f"Failed to load from PT file: {e}")
        logging.info("Trying to load from CSV file...")
        
        if embedding_type == 'tfidf':
            file_name = "tfidf_2.csv"
        else:
            file_name = f"{embedding_type}_2.csv"
            
        csv_path = f"../datasets/interim/embeddings/csv/{file_name}"
        df = pd.read_csv(csv_path)
        
        # For TF-IDF, check if we need to rename the label column
        if embedding_type == 'tfidf' and 'CommentClass_en' in df.columns:
            df.rename(columns={'CommentClass_en': 'labels'}, inplace=True)
            
        logging.info(f"Successfully loaded {embedding_type} embeddings from CSV: {csv_path}")
    
    # Ensure labels are in the correct format
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(safe_parse)
    else:
        # Try to find alternative label column 
        label_candidates = ['CommentClass_en', 'label', 'classes', 'class']
        for col in label_candidates:
            if col in df.columns:
                df.rename(columns={col: 'labels'}, inplace=True)
                df['labels'] = df['labels'].apply(safe_parse)
                break
        else:
            logging.error(f"No label column found in {embedding_type} embeddings")
            raise ValueError(f"No label column found in {embedding_type} embeddings")
    
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
    feature_cols = [col for col in df.columns if col not in ['original_index', 'labels', 'Unnamed: 0']]
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


def train_model(model, X_train, y_train, X_test, y_test, model_name, embedding_type,
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
                torch.save(model.state_dict(), f"../models/best_{embedding_type}_{model_name}_classifier.pt")
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
    model.load_state_dict(torch.load(f"../models/best_{embedding_type}_{model_name}_classifier.pt"))
    
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


def plot_training_curves(train_losses, val_losses, model_name, embedding_type):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_name} ({embedding_type.upper()})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../results/{embedding_type}_{model_name}_loss_curve.png")
    plt.close()


def plot_metrics_comparison(results_dict):
    """Plot performance metrics comparison between models and embedding types"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data for plotting
    data = []
    for embedding_type, models_data in results_dict.items():
        for model_name, metrics_dict in models_data.items():
            model_label = f"{model_name} ({embedding_type.upper()})"
            for metric in metrics:
                data.append({
                    'Model': model_label,
                    'Metric': metric.capitalize(),
                    'Value': metrics_dict[metric]
                })
    
    df_plot = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_plot)
    plt.title('Performance Metrics Comparison: CBOW vs Skip-gram vs TF-IDF')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("../results/embedding_comparison.png")
    plt.close()


def plot_embedding_comparison(results_dict):
    """Plot comparison of embedding types across metrics and models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(next(iter(results_dict.values())).keys())
    
    # For each metric, compare embeddings across models
    for metric in metrics:
        plt.figure(figsize=(12, 7))
        
        # Prepare data
        x = np.arange(len(model_names))
        width = 0.25  # Width of the bars
        
        # Create the bars for each embedding type
        embedding_types = list(results_dict.keys())
        for i, emb_type in enumerate(embedding_types):
            values = [results_dict[emb_type][model][metric] for model in model_names]
            offset = width * (i - len(embedding_types)/2 + 0.5)
            plt.bar(x + offset, values, width, label=emb_type.upper())
        
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison: CBOW vs Skip-gram vs TF-IDF')
        plt.xticks(x, model_names)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(f"../results/{metric}_comparison.png")
        plt.close()


# === 5. Main Function ===

def process_embedding_type(embedding_type):
    """Process a specific embedding type (CBOW, Skip-gram, or TF-IDF)"""
    logging.info(f"\n{'-'*70}\nProcessing {embedding_type.upper()} embeddings\n{'-'*70}")
    
    # Load embeddings
    df = load_embeddings(embedding_type)
    logging.info(f"Loaded {embedding_type} dataframe with shape: {df.shape}")
    
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
    results = {}
    
    for model_info in models:
        name = model_info['name']
        model = model_info['model']
        
        logging.info(f"\n{'='*50}\nTraining {name} with {embedding_type.upper()}\n{'='*50}")
        
        trained_model, metrics, train_losses, val_losses, class_report = train_model(
            model, X_train, y_train, X_test, y_test, name, embedding_type
        )
        
        # Log results
        logging.info(f"Results for {name} with {embedding_type.upper()}:")
        for metric, value in metrics.items():
            logging.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot learning curves
        plot_training_curves(train_losses, val_losses, name, embedding_type)
        
        # Save detailed class report
        pd.DataFrame(class_report).transpose().to_csv(
            f"../results/{embedding_type}_{name}_class_report.csv"
        )
        
        # Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_names': classes,
            'metrics': metrics,
            'architecture': str(trained_model)
        }, f"../models/{embedding_type}_{name}_classifier.pt")
        
        # Store results
        results[name] = metrics
    
    return results


def main():
    """Main function to run the classification pipeline for all embedding types"""
    logging.info("Starting embedding classification comparing CBOW, Skip-gram, and TF-IDF")
    
    all_results = {}
    
    # Process CBOW embeddings
    all_results['cbow'] = process_embedding_type('cbow')
    
    # Process Skip-gram embeddings
    all_results['skipgram'] = process_embedding_type('skipgram')
    
    # Process TF-IDF embeddings
    all_results['tfidf'] = process_embedding_type('tfidf')
    
    # Compare models and embeddings
    plot_metrics_comparison(all_results)
    plot_embedding_comparison(all_results)
    
    # Save overall results
    results_df = pd.DataFrame({
        f"{model}_{emb_type}": metrics 
        for emb_type, models in all_results.items() 
        for model, metrics in models.items()
    })
    results_df.to_csv("../results/embedding_comparison.csv")
    
    # Create a summary table
    summary_data = []
    for emb_type, models in all_results.items():
        for model_name, metrics in models.items():
            row = {
                'Embedding': emb_type.upper(),
                'Model': model_name
            }
            row.update({k.capitalize(): f"{v:.4f}" for k, v in metrics.items()})
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("../results/embedding_classification_summary.csv", index=False)
    print("\nClassification results summary:")
    print(summary_df.to_string(index=False))
    
    # Calculate average performance per embedding type
    print("\nAverage performance by embedding type:")
    for emb_type, models in all_results.items():
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            avg_metrics[metric] = np.mean([models[model][metric] for model in models])
        print(f"{emb_type.upper()}: " + ", ".join([f"{k.capitalize()}: {v:.4f}" for k, v in avg_metrics.items()]))
    
    logging.info("Classification comparison completed! Results saved to 'results' directory.")


if __name__ == "__main__":
    main()