import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_english_reviews(input_file, output_file):
    """Extract only English reviews from the dataset based on 'en_' prefix in ID"""
    logging.info(f"Loading Amazon Reviews dataset from {input_file}...")
    
    # Read the dataset
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logging.info(f"Total reviews in dataset: {total_rows}")
    
    # Determine which column contains the ID
    id_column = None
    possible_columns = ['id', 'ID', 'review_id', 'Unnamed: 0']
    
    for col in possible_columns:
        if col in df.columns:
            id_column = col
            break
    
    if not id_column:
        # If none of the expected columns exist, look for a column with values matching the pattern
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if values in this column have 'en_' prefix
                sample_values = df[col].dropna().head(10).astype(str)
                if any('en_' in str(val) for val in sample_values):
                    id_column = col
                    break
    
    if not id_column:
        logging.error("Could not find a suitable ID column in the dataset")
        return
    
    logging.info(f"Using '{id_column}' as the ID column for filtering English reviews")
    
    # Filter English reviews (those with 'en_' prefix in ID)
    english_df = df[df[id_column].astype(str).str.contains('en_')].copy()
    
    # Save to CSV
    logging.info(f"Found {len(english_df)} English reviews out of {total_rows}")
    logging.info(f"Saving English reviews to {output_file}")
    english_df.to_csv(output_file, index=False)
    
    logging.info("English reviews extraction completed successfully")
    return english_df

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "../datasets/exterim/amazonReviews.csv"  # Adjust path as needed
    output_file = "../datasets/interim/english_amazonReviews.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Extract English reviews
    extract_english_reviews(input_file, output_file)
