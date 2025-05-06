import os
import sys
import logging
import pandas as pd
import torch
from model1 import prepare_data, train_improved_model, evaluate_model
from Clustering import main as run_clustering

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main():
    """Run both SBERT classification and clustering analysis"""
    logging.info("Starting SBERT analysis pipeline...")
    
    # Step 1: Run the SBERT classification model
    logging.info("Step 1: Running SBERT classification model...")
    
    # Import and run the main function from model1.py
    from model1 import main as run_model1
    model1_results = run_model1()
    
    # Step 2: Run clustering and emotion analysis
    logging.info("Step 2: Running clustering and emotion analysis...")
    clustering_results = run_clustering()
    
    # Step 3: Generate combined report
    logging.info("Step 3: Generating combined report...")
    
    # Create a summary of both analyses
    report = {
        'classification_metrics': pd.read_csv('../results/sbert_metrics.csv').to_dict('records')[0],
        'optimal_clusters': len(clustering_results['cluster_analysis']),
        'cluster_purity': clustering_results['validation_results']['overall_purity'],
        'emotion_clusters': {
            cluster_id: summary['top_emotions'] 
            for cluster_id, summary in clustering_results['summaries'].items()
        }
    }
    
    # Save report as JSON
    import json
    with open('../results/combined_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    logging.info("Combined analysis completed! Report saved to '../results/combined_analysis_report.json'")
    
    return {
        'model1_results': model1_results,
        'clustering_results': clustering_results
    }

if __name__ == "__main__":
    main()
