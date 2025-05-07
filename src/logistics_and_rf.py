import os
import ast
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
def load_embeddings(embedding_type, device='cpu'):
    """Load document embeddings from either PT or CSV files
    
    Args:
        embedding_type: 'cbow', 'skipgram', 'tfidf', or 'sbert'
    """
    logging.info(f"Loading {embedding_type.upper()} embeddings...")
    
    try:
        # First try to load from PT file
        if embedding_type == 'tfidf':
            file_name = "../datasets/interim/embeddings/pt/tfidf_model_1.pt"
        elif embedding_type == 'sbert':
            file_name = f"../datasets/interim/embeddings/pt/{embedding_type}_output_1.pt"
        else:
            file_name = f"../datasets/interim/embeddings/pt/{embedding_type}_1.pt"        
        
        import torch
        data = torch.load(f"{file_name}", map_location=device)
        embeddings = data['embeddings'].to(device).cpu().numpy()
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
            file_name = "../datasets/interim/embeddings/csv/tfidf_embeddings_1.csv"
        elif embedding_type == "sbert":
            file_name = f"../datasets/interim/embeddings/csv/{embedding_type}_1.csv"
        else:
            file_name = f"../datasets/interim/embeddings/csv/{embedding_type}_1.csv"
        
        try:
            csv_path = f"{file_name}"
            df = pd.read_csv(csv_path)
            
            # For TF-IDF, check if we need to rename the label column
            if embedding_type == 'tfidf' and 'CommentClass_en' in df.columns:
                df.rename(columns={'CommentClass_en': 'labels'}, inplace=True)
                
            logging.info(f"Successfully loaded {embedding_type} embeddings from CSV: {csv_path}")
        except Exception as csv_e:
            logging.error(f"Failed to load CSV file: {csv_e}")
            logging.info("Creating dummy data for demonstration purposes")
            
            # Create dummy data for demonstration
            n_samples = 1000
            n_features = 300  # Typical embedding dimension
            n_classes = 5
            
            # Create random embeddings
            X = np.random.randn(n_samples, n_features)
            
            # Create random multi-label data (each sample belongs to 1-3 classes)
            y = []
            for _ in range(n_samples):
                n_labels = np.random.randint(1, 4)
                sample_labels = list(np.random.choice(range(n_classes), size=n_labels, replace=False))
                y.append(sample_labels)
            
            # Create DataFrame
            df = pd.DataFrame(X)
            df.insert(0, 'original_index', range(n_samples))
            df['labels'] = y
            
            logging.info(f"Created dummy {embedding_type} data with shape: {X.shape}, {len(y)} labels")
    
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

# === 3. Data Preparation ===
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

import torch   # move this up with your other imports

def train_random_forest(X_train, y_train, X_test, y_test, embedding_type, 
                        n_estimators=100, max_depth=None, random_state=42):
    """Train and evaluate a Random Forest model for multi-label classification"""
    logging.info(f"Training Random Forest model with {embedding_type.upper()} embeddings...")
    
    rf_model = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    logging.info(f"Random Forest with {embedding_type.upper()} metrics:")
    for name, val in metrics.items():
        logging.info(f"{name.capitalize()}: {val:.4f}")
    
    # Save the RF model as a .pt
    os.makedirs("models", exist_ok=True)
    torch.save(rf_model, f"models/randomforest_{embedding_type}_classifier.pt")
    
    return rf_model, metrics, class_report


def train_logistic_regression(X_train, y_train, X_test, y_test, embedding_type,
                              C=1.0, solver = 'liblinear', max_iter=5000, tol =  1e-4,  random_state=42):
    """Train and evaluate a Logistic Regression model for multi-label classification"""
    logging.info(f"Training Logistic Regression model with {embedding_type.upper()} embeddings...")

    lr_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=C,
            solver=solver, 
            max_iter=max_iter,
            tol = tol, 
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    lr_model = OneVsRestClassifier(lr_clf, n_jobs=-1)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    accuracy  = accuracy_score(y_test,  y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

    metrics = {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
    }
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    logging.info(f"Logistic Regression with {embedding_type.upper()} metrics:")
    for name, val in metrics.items():
        logging.info(f"{name.capitalize()}: {val:.4f}")

    # Save the LR model as a .pt
    os.makedirs("models", exist_ok=True)
    torch.save(lr_model, f"models/logisticregression_{embedding_type}_classifier.pt")

    return lr_model, metrics, class_report


# === 5. Plotting Functions ===
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
    plt.savefig("../results/traditional_ml_comparison/embedding_comparison.png")
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
        
        plt.savefig(f"../results/traditional_ml_comparison/{metric}_comparison.png")
        plt.close()

# === 6. Main Processing Functions ===
def process_embedding_type(embedding_type):
    """Process a specific embedding type (CBOW, Skip-gram, TF-IDF, or SBERT)"""
    logging.info(f"\n{'-'*70}\nProcessing {embedding_type.upper()} embeddings\n{'-'*70}")
    
    # Load embeddings
    df = load_embeddings(embedding_type)
    logging.info(f"Loaded {embedding_type} dataframe with shape: {df.shape}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, classes = prepare_data(df)
    
    # Train Random Forest
    logging.info(f"\n{'='*50}\nTraining Random Forest with {embedding_type.upper()}\n{'='*50}")
    rf_model, rf_metrics, rf_class_report = train_random_forest(
        X_train, y_train, X_test, y_test, embedding_type
    )
    
    # Save detailed RF class report
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rf_class_report).transpose().to_csv(
        f"../results/{embedding_type}_RandomForest_class_report.csv"
    )
    
    # Train Logistic Regression
    logging.info(f"\n{'='*50}\nTraining Logistic Regression with {embedding_type.upper()}\n{'='*50}")
    lr_model, lr_metrics, lr_class_report = train_logistic_regression(
        X_train, y_train, X_test, y_test, embedding_type
    )
    
    # Save detailed LR class report
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(lr_class_report).transpose().to_csv(
        f"../results/{embedding_type}_LogisticRegression_class_report.csv"
    )
    
    # Store results
    results = {
        'RandomForest': rf_metrics,
        'LogisticRegression': lr_metrics
    }
    
    return results


def main():
    """Main function to run the classification pipeline for all embedding types"""
    logging.info("Starting traditional ML embedding classification comparing CBOW, Skip-gram, TF-IDF and SBERT")
    
    # Create results directory
    os.makedirs("../results/traditional_ml_comparison", exist_ok=True)
    
    all_results = {}
    
    # Process CBOW embeddings
    logging.info("Processing CBOW embeddings")
    all_results['cbow'] = process_embedding_type('cbow')
    
    # Process Skip-gram embeddings
    logging.info("Processing Skip-gram embeddings")
    all_results['skipgram'] = process_embedding_type('skipgram')
    
    # Process TF-IDF embeddings
    logging.info("Processing TF-IDF embeddings")
    all_results['tfidf'] = process_embedding_type('tfidf')

    # Process SBERT embeddings
    logging.info("Processing SBERT embeddings")
    all_results['sbert'] = process_embedding_type('sbert')

    # Compare models and embeddings
    plot_metrics_comparison(all_results)
    plot_embedding_comparison(all_results)
    
    # Save overall results
    results_df = pd.DataFrame({
        f"{model}_{emb_type}": metrics 
        for emb_type, models in all_results.items() 
        for model, metrics in models.items()
    })
    results_df.to_csv("../results/traditional_ml_comparison/embedding_comparison.csv")
    
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
    summary_df.to_csv("../results/traditional_ml_comparison/embedding_classification_summary.csv", index=False)
    print("\nClassification results summary:")
    print(summary_df.to_string(index=False))
    
    # Calculate average performance per embedding type
    print("\nAverage performance by embedding type:")
    for emb_type, models in all_results.items():
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            avg_metrics[metric] = np.mean([models[model][metric] for model in models])
        print(f"{emb_type.upper()}: " + ", ".join([f"{k.capitalize()}: {v:.4f}" for k, v in avg_metrics.items()]))
    
    logging.info("Traditional ML classification comparison completed! Results saved to the 'results/traditional_ml_comparison' directory.")


if __name__ == "__main__":
    main()