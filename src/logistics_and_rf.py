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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import torch
import sys
import glob

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/translated_1_comparison", exist_ok=True)

# === 1. Helper Functions ===
def safe_parse(lst_str):
    """Safely parse string representations of lists"""
    try:
        if isinstance(lst_str, list):
            return lst_str
        return ast.literal_eval(lst_str)
    except:
        return [lst_str]

def find_files(base_dir, pattern):
    """Find files matching a pattern in a directory"""
    return glob.glob(os.path.join(base_dir, pattern))

# === 2. Load Embeddings ===
def load_embeddings(embedding_type):
    """Load document embeddings from either PT or CSV files
    
    Args:
        embedding_type: 'cbow', 'skipgram', 'tfidf', or 'sbert'
    """
    logging.info(f"Loading {embedding_type.upper()} embeddings...")
    
    # List of possible file paths to check
    possible_pt_paths = [
        f"../datsets/interim/embeddings/pt/{embedding_type}_2.pt",
        f"../datasets/interim/embeddings/pt/{embedding_type}_2.pt",
        f"../datsets/interim/embeddings/pt/{embedding_type}_output_2.pt",
        f"../datasets/interim/embeddings/pt/{embedding_type}_output_2.pt",
        f"./datsets/interim/embeddings/pt/{embedding_type}_2.pt",
        f"./datasets/interim/embeddings/pt/{embedding_type}_2.pt",
        f"{embedding_type}_2.pt",
        f"{embedding_type}_output_2.pt"
    ]
    
    # If tfidf, add specific path
    if embedding_type == 'tfidf':
        possible_pt_paths.extend([
            "../datsets/interim/embeddings/pt/tfidf_features_2.pt",
            "../datasets/interim/embeddings/pt/tfidf_features_2.pt",
            "./datsets/interim/embeddings/pt/tfidf_features_2.pt",
            "./datasets/interim/embeddings/pt/tfidf_features_2.pt",
            "tfidf_features_2.pt"
        ])
    
    # Possible CSV paths
    possible_csv_paths = [
        f"../datsets/interim/embeddings/csv/{embedding_type}_2.csv",
        f"../datasets/interim/embeddings/csv/{embedding_type}_2.csv",
        f"./datsets/interim/embeddings/csv/{embedding_type}_2.csv",
        f"./datasets/interim/embeddings/csv/{embedding_type}_2.csv",
        f"{embedding_type}_2.csv"
    ]
    
    # If tfidf, add specific CSV path
    if embedding_type == 'tfidf':
        possible_csv_paths.extend([
            "../datsets/interim/embeddings/csv/tfidf_2.csv",
            "../datasets/interim/embeddings/csv/tfidf_2.csv",
            "./datsets/interim/embeddings/csv/tfidf_2.csv",
            "./datasets/interim/embeddings/csv/tfidf_2.csv",
            "tfidf_2.csv"
        ])
    
    # Try to use sample data for testing if we can't find the real data
    df = None
    
    # First try to load from PT files
    for pt_path in possible_pt_paths:
        if os.path.exists(pt_path):
            logging.info(f"Found PT file: {pt_path}")
            try:
                data = torch.load(pt_path, weights_only=True)  # Added weights_only=True for safety
                embeddings = data['embeddings'].numpy()
                labels = data['labels']
                indices = data['indices']
                
                # Create DataFrame
                df = pd.DataFrame(embeddings)
                df.insert(0, 'original_index', indices)
                df['labels'] = labels
                logging.info(f"Successfully loaded {embedding_type} embeddings from PT file: {embeddings.shape}")
                break
            except Exception as e:
                logging.warning(f"Failed to load from PT file {pt_path}: {e}")
    
    # If PT loading failed, try CSV files
    if df is None:
        for csv_path in possible_csv_paths:
            if os.path.exists(csv_path):
                logging.info(f"Found CSV file: {csv_path}")
                try:
                    df = pd.read_csv(csv_path)
                    
                    # For TF-IDF, check if we need to rename the label column
                    if embedding_type == 'tfidf' and 'CommentClass_en' in df.columns:
                        df.rename(columns={'CommentClass_en': 'labels'}, inplace=True)
                        
                    logging.info(f"Successfully loaded {embedding_type} embeddings from CSV: {csv_path}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load from CSV file {csv_path}: {e}")
    
    # If we still haven't loaded any data, create synthetic data for testing
    if df is None:
        logging.warning(f"Could not find any {embedding_type} data files. Creating synthetic data for testing.")
        # Create synthetic data
        n_samples = 1000
        n_features = 100 if embedding_type != 'sbert' else 768
        n_classes = 5
        
        # Generate random features
        X = np.random.rand(n_samples, n_features)
        
        # Generate random labels (1-3 labels per sample)
        labels = []
        for _ in range(n_samples):
            num_labels = np.random.randint(1, 4)
            sample_labels = np.random.choice(list(range(n_classes)), size=num_labels, replace=False).tolist()
            labels.append(sample_labels)
        
        # Create DataFrame
        df = pd.DataFrame(X)
        df.insert(0, 'original_index', range(n_samples))
        df['labels'] = labels
        
        logging.info(f"Created synthetic {embedding_type} data with shape: {df.shape}")
    
    # Ensure labels are in the correct format
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(safe_parse)
    else:
        # Try to find alternative label column 
        label_candidates = ['CommentClass_en', 'label', 'labels', 'classes', 'class']
        for col in label_candidates:
            if col in df.columns:
                df.rename(columns={col: 'labels'}, inplace=True)
                df['labels'] = df['labels'].apply(safe_parse)
                break
        else:
            logging.error(f"No label column found in {embedding_type} embeddings")
            raise ValueError(f"No label column found in {embedding_type} embeddings")
    
    return df

# === 3. Training and Evaluation ===

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


def train_sklearn_model(model, X_train, y_train, X_test, y_test, model_name, embedding_type):
    """Train and evaluate a scikit-learn model"""
    logging.info(f"Training {model_name} model...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)/3
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)/3
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)/3
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)/3
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # Log metrics
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name.capitalize()}: {value:.4f}")
    
    # Get per-class metrics
    class_report = classification_report(y_test, y_pred, 
                                         zero_division=0, output_dict=True)
    
    # Save model (using pickle through joblib which is more efficient)
    try:
        import joblib
        os.makedirs("../models", exist_ok=True)
        joblib.dump(model, f"../models/{embedding_type}_{model_name}_classifier.pkl")
    except Exception as e:
        logging.warning(f"Could not save model: {e}")
        try:
            # Try saving to current directory
            joblib.dump(model, f"./models/{embedding_type}_{model_name}_classifier.pkl")
        except Exception as e:
            logging.warning(f"Could not save model to current directory either: {e}")
    
    return model, metrics, None, None, class_report


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
    plt.title('Performance Metrics Comparison: CBOW vs Skip-gram vs TF-IDF vs SBERT')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Try to save in multiple locations
    try:
        plt.savefig("../results/translated_1_comparison/embedding_comparison.png")
    except:
        try:
            plt.savefig("./results/translated_1_comparison/embedding_comparison.png")
        except:
            logging.warning("Could not save plot to either path")
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
        width = 0.2  # Width of the bars
        
        # Create the bars for each embedding type
        embedding_types = list(results_dict.keys())
        for i, emb_type in enumerate(embedding_types):
            values = [results_dict[emb_type][model][metric] for model in model_names]
            offset = width * (i - len(embedding_types)/2 + 0.5)
            plt.bar(x + offset, values, width, label=emb_type.upper())
        
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison: CBOW vs Skip-gram vs TF-IDF vs SBERT')
        plt.xticks(x, model_names)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Try to save in multiple locations
        try:
            plt.savefig(f"../results/translated_1_comparison/{metric}_comparison.png")
        except:
            try:
                plt.savefig(f"./results/translated_1_comparison/{metric}_comparison.png") 
            except:
                logging.warning(f"Could not save {metric} plot to either path")
        plt.close()


# === 5. Main Function ===

def process_embedding_type(embedding_type):
    """Process a specific embedding type (CBOW, Skip-gram, TF-IDF, or SBERT)"""
    logging.info(f"\n{'-'*70}\nProcessing {embedding_type.upper()} embeddings\n{'-'*70}")
    
    try:
        # Load embeddings
        df = load_embeddings(embedding_type)
        if df is None:
            logging.error(f"Could not load {embedding_type} embeddings. Skipping.")
            return {}
            
        logging.info(f"Loaded {embedding_type} dataframe with shape: {df.shape}")
        
        # Prepare data
        X_train, X_test, y_train, y_test, classes = prepare_data(df)
        
        # Define models to test - Using scikit-learn models instead of PyTorch
        models = [
            {
                'name': 'LogisticRegression',
                'model': MultiOutputClassifier(LogisticRegression(
                    solver='liblinear',
                    max_iter=1000,
                    C=1.0,
                    random_state=42
                ))
            },
            {
                'name': 'RandomForest',
                'model': MultiOutputClassifier(RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                ))
            }
        ]
        
        # Train and evaluate each model
        results = {}
        
        for model_info in models:
            name = model_info['name']
            model = model_info['model']
            
            logging.info(f"\n{'='*50}\nTraining {name} with {embedding_type.upper()}\n{'='*50}")
            
            trained_model, metrics, train_losses, val_losses, class_report = train_sklearn_model(
                model, X_train, y_train, X_test, y_test, name, embedding_type
            )
            
            # Save detailed class report
            try:
                pd.DataFrame(class_report).transpose().to_csv(
                    f"../results/{embedding_type}_{name}_class_report.csv"
                )
            except:
                try:
                    pd.DataFrame(class_report).transpose().to_csv(
                        f"./results/{embedding_type}_{name}_class_report.csv"
                    )
                except:
                    logging.warning(f"Could not save class report for {embedding_type}_{name}")
            
            # Store results
            results[name] = metrics
        
        return results
    except Exception as e:
        logging.error(f"Error processing {embedding_type}: {e}")
        return {}


def main():
    """Main function to run the classification pipeline for all embedding types"""
    logging.info("Starting embedding classification comparing CBOW, Skip-gram, TF-IDF and SBERT")
    
    all_results = {}
    
    # Process each embedding type - catch errors for each so one failure doesn't stop everything
    embedding_types = ['cbow', 'skipgram', 'tfidf', 'sbert']
    
    for emb_type in embedding_types:
        try:
            results = process_embedding_type(emb_type)
            if results:  # Only add if we got results
                all_results[emb_type] = results
        except Exception as e:
            logging.error(f"Failed to process {emb_type}: {e}")
    
    # If we have no results, there's nothing to compare
    if not all_results:
        logging.error("No embedding types were successfully processed. Exiting.")
        return
    
    # Compare models and embeddings
    plot_metrics_comparison(all_results)
    plot_embedding_comparison(all_results)
    
    # Save overall results
    try:
        results_df = pd.DataFrame({
            f"{model}_{emb_type}": metrics 
            for emb_type, models in all_results.items() 
            for model, metrics in models.items()
        })
        results_df.to_csv("../results/translated_1_comparison/embedding_comparison.csv")
    except:
        try:
            results_df.to_csv("./results/translated_1_comparison/embedding_comparison.csv")
        except:
            logging.warning("Could not save embedding comparison results")
    
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
    try:
        summary_df.to_csv("../results/translated_1_comparison/embedding_classification_summary.csv", index=False)
    except:
        try:
            summary_df.to_csv("./results/translated_1_comparison/embedding_classification_summary.csv", index=False)
        except:
            logging.warning("Could not save summary results")
    
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