import pandas as pd
import glob
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1) Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2) Prepare text‐processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text: str):
    # Lowercase
    text = str(text).lower()
    # Remove URLs and non‐alphabetic chars
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords & short tokens, then lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return clean_tokens

def safe_parse(label_str: str):
    try:
        return ast.literal_eval(label_str)
    except Exception:
        # Fallback to single‐item list
        return [label_str]

# 3) Process each translated_output CSV
for path in glob.glob('../datasets/interim/converted_amazonReviews_50k.csv'):
    df = pd.read_csv(path)
    
    # Drop any Unnamed columns (pandas index leftovers)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    
    # Ensure Comment_en exists and is a string
    df['Comment_en'] = df['Comment_en'].fillna('').astype(str)
    
    # Parse the stringified label lists
    df['labels'] = df['CommentClass_en'].fillna('[]').apply(safe_parse)
    
    # Clean & tokenize text
    df['tokens'] = df['Comment_en'].apply(clean_and_tokenize)
    
    # Join tokens back into a cleaned sentence (optional)
    df['clean_comment'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    
    # Save cleaned version
    clean_path = path.replace('.csv', '_clean.csv')
    df.to_csv(clean_path, index=False)
    print(f"→ Saved cleaned dataset: {clean_path}")
