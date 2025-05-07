import pandas as pd
import time
import ast
from deep_translator import GoogleTranslator
import os

def translate_dataset(input_file, output_file, batch_size=10):
    # Read the CSV file
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    # Check if output file exists
    file_exists = os.path.isfile(output_file)
    
    # Process in batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Create empty lists for translated content
        translated_comments = []
        translated_labels = []
        
        # Translate comments and labels for this batch
        for index, row in batch_df.iterrows():
            try:
                # Translate comment
                if 'Comment' in df.columns:
                    comment_text = str(row['Comment'])
                    # Google Translator has a character limit, so we need to handle long texts
                    if len(comment_text) > 5000:
                        # Split into chunks of 5000 characters
                        chunks = [comment_text[i:i+5000] for i in range(0, len(comment_text), 5000)]
                        translated_chunks = []
                        for chunk in chunks:
                            translated_chunk = GoogleTranslator(source='tr', target='en').translate(chunk)
                            translated_chunks.append(translated_chunk)
                        translated_comment = ''.join(translated_chunks)
                    else:
                        translated_comment = GoogleTranslator(source='tr', target='en').translate(comment_text)
                    translated_comments.append(translated_comment)
                
                # Translate label
                if 'CommentClass' in df.columns:
                    try:
                        label_list = ast.literal_eval(str(row['CommentClass']))
                        # Translate each label in the list
                        translated_label_list = []
                        for label in label_list:
                            translated_label = GoogleTranslator(source='tr', target='en').translate(label)
                            translated_label_list.append(translated_label)
                        # Convert back to string representation of list
                        translated_labels.append(str(translated_label_list))
                    except:
                        # If parsing fails, translate as is
                        translated_label = GoogleTranslator(source='tr', target='en').translate(str(row['CommentClass']))
                        translated_labels.append(translated_label)
                
                # Add delay to avoid hitting API limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error at row {index}: {str(e)}")
                translated_comments.append(str(row['Comment']) if 'Comment' in df.columns else '')
                translated_labels.append(str(row['CommentClass']) if 'CommentClass' in df.columns else '')
        
        # Update batch DataFrame with translated content
        if 'Comment' in df.columns:
            batch_df['Comment_en'] = translated_comments
        if 'CommentClass' in df.columns:
            batch_df['CommentClass_en'] = translated_labels
        
        # Append to output file
        if file_exists and start_idx > 0:
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(output_file, mode='w', index=False)
            file_exists = True
        
        print(f"Processed and saved batch {start_idx//batch_size + 1}/{(total_rows-1)//batch_size + 1} ({end_idx}/{total_rows} rows)")

# Example usage
# translate_dataset('electronicComments.csv', 'translated_output_1.csv', batch_size=100)
# translate_dataset('homeComments.csv', 'translated_output_2.csv', batch_size=100)
translate_dataset(
    '../datasets/interim/',
    '../datasets/interim/translated_output_3.csv',
    batch_size=100
)

