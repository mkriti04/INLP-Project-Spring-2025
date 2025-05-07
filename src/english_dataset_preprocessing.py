# import pandas as pd
# import os
# import logging
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import NMF
# import numpy as np
# from collections import defaultdict

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def extract_topics_from_dataset(df, n_topics=10, n_top_words=20):
#     """
#     Extract topics from the dataset using NMF (Non-negative Matrix Factorization)
#     and TF-IDF vectorization
#     """
#     logging.info("Extracting topics from dataset...")
    
#     # Combine title and review body if available
#     if 'review_title' in df.columns:
#         corpus = df['review_title'].fillna('') + ' ' + df['review_body'].fillna('')
#     else:
#         corpus = df['review_body'].fillna('')
    
#     # Create TF-IDF vectorizer
#     tfidf_vectorizer = TfidfVectorizer(
#         max_df=0.95, 
#         min_df=2,
#         max_features=1000,
#         stop_words='english'
#     )
    
#     # Fit and transform the corpus
#     tfidf = tfidf_vectorizer.fit_transform(corpus)
    
#     # Extract feature names
#     feature_names = tfidf_vectorizer.get_feature_names_out()
    
#     # Apply NMF
#     nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    
#     # Extract topics
#     topics = {}
#     for topic_idx, topic in enumerate(nmf.components_):
#         top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
#         top_words = [feature_names[i] for i in top_words_idx]
#         topics[f"Topic_{topic_idx}"] = top_words
#         logging.info(f"Topic {topic_idx}: {', '.join(top_words[:10])}")
    
#     return topics, nmf, tfidf_vectorizer

# def map_topics_to_classes(topics):
#     """
#     Map extracted topics to predefined classes based on semantic similarity
#     """
#     # Define base classes and their seed keywords
#     base_classes = {
#         'Product': ['quality', 'design', 'material', 'product', 'item', 'works', 'feature'],
#         'Priceperformance': ['price', 'value', 'worth', 'expensive', 'cheap', 'cost', 'money'],
#         'Shipping': ['shipping', 'delivery', 'arrived', 'package', 'box'],
#         'CustomerService': ['service', 'support', 'customer', 'return', 'refund', 'warranty'],
#         "ProductQuality": [
#     "Pure Grace",
#     "Smell",
#     "Fake",
#     "Stay away from this book",
#     "disappointing",
#     "bad",
#     "no",
#     "good",
#     "Item",
#     "Fast delivery",
#     "expensive",
#     "cheap",
#     "quality",
#     "durable",
#     "well-made",
#     "material",
#     "construction",
#     "Had",
#     "Bought",
#     "Worst",
#     "Unacceptable",
#     "Unpleasant",
#     "uncomfortable",
#     "unreliable",
#     "Battle of the Brand",
#     "Has",
#     "It",
#     "shape",
#     "Urgent",
#     "Small",
#     "Great",
#     "Alarm",
#     "Sleek",
#     "Unprofessional"
#   ],
#         "Customer rating": [
#     "1",
#     "out",
#     "of",
#     "5",
#     "0"
#   ],
#   "ProductFailure": [
#     "Unusual",
#     "Worst",
#     "Really Stupid",
#     "Dried out",
#     "Wrong part"
#   ],
#   "ProductSupplier": [
#     "Item",
#     "ProductSupplier",
#     "Price"
#   ],
#   "ProductSize": [
#     "13",
#     "inch",
#     "half",
#     "extra",
#     "space",
#     "length",
#     "width",
#     "accurate"
#   ],
#   "ProductFactory": [
#     "Color",
#     "Most",
#     "Odd",
#     "Smell",
#     "Stylized",
#     "Size",
#     "Sizing"
#   ],
#   "ProductPrime": [
#     "Good",
#     "No"
#   ],
#   "ProductType": [
#     "Uncomfortable",
#     "Leaky",
#     "Like",
#     "Case",
#     "ProductSize",
#     "ProductType",
#     "Watered",
#     "down",
#     "ProductFailure",
#     "Typical",
#     "ProductRating",
#     "Not",
#     "Real",
#     "1",
#     "Star",
#     "Stone",
#     "Metal",
#     "Wood",
#     "Cheap",
#     "Smell",
#     "Paint",
#     "Chipped",
#     "Unusable"
#   ],
#   "Quality": [
#     "Small",
#     "bad",
#     "convenient",
#     "defective",
#     "waterproof",
#     "unlikely",
#     "unusual"
#   ],
#   "Product Quality": [
#     "No",
#     "Great",
#     "Ideal",
#     "Unacceptable",
#     "Defective",
#     "Disappointing",
#     "Hybrid",
#     "Small",
#     "Worst",
#     "Clean",
#     "Dirty",
#     "Stylized",
#     "Sleepsizzing",
#     "Sleek",
#     "Nook",
#     "Product",
#     "Quality",
#     "Good",
#     "Fast",
#     "Hyper",
#     "Duty",
#     "Important",
#     "Bought",
#     "Third",
#     "Offensive",
#     "NONE"
#   ],
#   "ProductOptions": [
#     "Correct",
#     "Unacceptable",
#     "Waiting",
#     "Overpriced",
#     "Not Camera"
#   ],
#   "ProductWords": [
#     "Wrong"
#   ],
#   "Size": [
#     "XL",
#     "Adult",
#     "Size",
#     "Child"
#   ],
#   "Damage": [
#     "Fast delivery",
#     "Battle",
#     "Tasty",
#     "Attack",
#     "damage",
#     "repair",
#     "damage",
#     "warranty",
#     "Very poor",
#     "damaged"
#   ]
#     }
    
#     # Map topics to classes
#     topic_to_class = {}
#     class_keywords = defaultdict(set)
    
#     for topic_name, topic_words in topics.items():
#         best_match = None
#         best_score = 0
        
#         for class_name, seed_words in base_classes.items():
#             # Count overlapping words
#             overlap = set(topic_words).intersection(set(seed_words))
#             score = len(overlap)
            
#             if score > best_score:
#                 best_score = score
#                 best_match = class_name
        
#         # If no good match, assign to "Other"
#         if best_score == 0:
#             best_match = "Other"
        
#         topic_to_class[topic_name] = best_match
        
#         # Add topic words to class keywords
#         class_keywords[best_match].update(topic_words)
    
#     # Convert sets to lists
#     for class_name in class_keywords:
#         class_keywords[class_name] = list(class_keywords[class_name])
    
#     return topic_to_class, dict(class_keywords)

# def classify_review(review_text, review_title, class_keywords):
#     """
#     Classify a review into one or more classes based on expanded keywords
#     """
#     full_text = f"{review_title} {review_text}" if review_title else review_text
#     full_text = full_text.lower()
    
#     classes = []
    
#     # Check for keywords in the text
#     for class_name, keywords in class_keywords.items():
#         for keyword in keywords:
#             if re.search(r'\b' + re.escape(keyword) + r'\b', full_text):
#                 classes.append(class_name)
#                 break
    
#     # If no classes were found, default to 'Product'
#     if not classes or (len(classes) == 1 and classes[0] == 'Other'):
#         classes = ['Product']
#     elif 'Other' in classes and len(classes) > 1:
#         classes.remove('Other')
    
#     return classes

# def convert_amazon_reviews_format(input_file, output_file):
#     """Convert Amazon reviews to the format matching the Turkish dataset"""
#     logging.info(f"Loading Amazon Reviews dataset from {input_file}...")
    
#     # Read the dataset
#     df = pd.read_csv(input_file)
#     logging.info(f"Loaded {len(df)} reviews")
    
#     # Extract topics from the dataset
#     topics, _, _ = extract_topics_from_dataset(df)
    
#     # Map topics to classes and get expanded keywords
#     topic_to_class, class_keywords = map_topics_to_classes(topics)
    
#     logging.info("Expanded class keywords:")
#     for class_name, keywords in class_keywords.items():
#         logging.info(f"{class_name}: {', '.join(keywords[:10])}...")
    
#     # Create a new DataFrame with the required format
#     new_df = pd.DataFrame()
    
#     # Map review_body to Comment_en
#     new_df['Comment_en'] = df['review_body']
    
#     # Extract CommentClass_en based on review content with expanded keywords
#     logging.info("Classifying reviews with expanded keywords...")
#     new_df['CommentClass_en'] = df.apply(
#         lambda row: classify_review(
#             row['review_body'], 
#             row.get('review_title', ''), 
#             class_keywords
#         ), 
#         axis=1
#     )
    
#     # Use the index as Unnamed: 0
#     new_df['Unnamed: 0'] = df.index
    
#     # Save to CSV
#     logging.info(f"Saving converted format to {output_file}")
#     new_df.to_csv(output_file, index=False)
    
#     logging.info("Conversion completed successfully")
#     return new_df

# if __name__ == "__main__":
#     # Define input and output file paths
#     input_file = "../datasets/exterim/english_amazonReviews.csv"  # Adjust path as needed
#     output_file = "../datasets/interim/converted_amazonReviews.csv"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     # Convert reviews format
#     convert_amazon_reviews_format(input_file, output_file)



import pandas as pd
import os
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from collections import defaultdict
from nltk.stem import PorterStemmer
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_topics_from_dataset(df, n_topics=10, n_top_words=20):
    """
    Extract topics from the dataset using NMF (Non-negative Matrix Factorization)
    and TF-IDF vectorization
    """
    logging.info("Extracting topics from dataset...")
    
    # Combine title and review body if available
    if 'review_title' in df.columns:
        corpus = df['review_title'].fillna('') + ' ' + df['review_body'].fillna('')
    else:
        corpus = df['review_body'].fillna('')
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, 
        min_df=2,
        max_features=1000,
        stop_words='english'
    )
    
    # Fit and transform the corpus
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    
    # Extract feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Apply NMF
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    
    # Extract topics
    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic_{topic_idx}"] = top_words
        logging.info(f"Topic {topic_idx}: {', '.join(top_words[:10])}")
    
    return topics, nmf, tfidf_vectorizer

def preprocess_keywords(base_classes):
    """
    Preprocess keywords to improve matching:
    1. Break multi-word phrases into individual words
    2. Apply stemming to handle word variations
    3. Remove duplicates
    """
    stemmer = PorterStemmer()
    processed_classes = {}
    
    for class_name, keywords in base_classes.items():
        processed_keywords = set()
        
        for keyword in keywords:
            # Add the original keyword (lowercase)
            processed_keywords.add(keyword.lower())
            
            # For multi-word phrases, add individual words
            if ' ' in keyword:
                words = keyword.lower().split()
                for word in words:
                    # Skip very short words and common words
                    if len(word) > 2 and word not in ['the', 'and', 'for', 'this', 'that', 'with', 'from']:
                        processed_keywords.add(word)
            
            # Add stemmed version
            stemmed = stemmer.stem(keyword.lower())
            if stemmed != keyword.lower():
                processed_keywords.add(stemmed)
        
        processed_classes[class_name] = list(processed_keywords)
    
    return processed_classes

def map_topics_to_classes(topics):
    """
    Map extracted topics to predefined classes based on semantic similarity
    with improved matching for better class distribution
    """
    # Define base classes and their seed keywords
    base_classes = {
        'Product': ['quality', 'design', 'material', 'product', 'item', 'works', 'feature'],
        'Priceperformance': ['price', 'value', 'worth', 'expensive', 'cheap', 'cost', 'money'],
        'Shipping': ['shipping', 'delivery', 'arrived', 'package', 'box'],
        'CustomerService': ['service', 'support', 'customer', 'return', 'refund', 'warranty'],
        "ProductQuality": [
            "Pure Grace", "Smell", "Fake", "Stay away from this book", "disappointing",
            "bad", "no", "good", "Item", "Fast delivery", "expensive", "cheap",
            "quality", "durable", "well-made", "material", "construction", "Had",
            "Bought", "Worst", "Unacceptable", "Unpleasant", "uncomfortable", 
            "unreliable", "Battle of the Brand", "Has", "It", "shape", "Urgent",
            "Small", "Great", "Alarm", "Sleek", "Unprofessional"
        ],
        "Customer rating": ["1", "out", "of", "5", "0", "stars", "rating", "rated", "star"],
        "ProductFailure": [
            "Unusual", "Worst", "Really Stupid", "Dried out", "Wrong part", "broken",
            "failed", "failure", "stopped working", "defect", "malfunction"
        ],
        "ProductSupplier": ["Item", "ProductSupplier", "Price", "seller", "vendor", "supplier", "manufacturer"],
        "ProductSize": [
            "13", "inch", "half", "extra", "space", "length", "width", "accurate",
            "size", "dimension", "fit", "large", "small", "medium"
        ],
        "ProductFactory": [
            "Color", "Most", "Odd", "Smell", "Stylized", "Size", "Sizing",
            "factory", "made", "manufacturing", "production"
        ],
        "ProductPrime": ["Good", "No", "prime", "delivery", "shipping", "fast"],
        "ProductType": [
            "Uncomfortable", "Leaky", "Like", "Case", "ProductSize", "ProductType",
            "Watered", "down", "ProductFailure", "Typical", "ProductRating", "Not",
            "Real", "1", "Star", "Stone", "Metal", "Wood", "Cheap", "Smell", "Paint",
            "Chipped", "Unusable", "type", "model", "version", "edition"
        ],
        "Quality": [
            "Small", "bad", "convenient", "defective", "waterproof", "unlikely", "unusual",
            "poor", "excellent", "superior", "inferior", "standard"
        ],
        "Product Quality": [
            "No", "Great", "Ideal", "Unacceptable", "Defective", "Disappointing",
            "Hybrid", "Small", "Worst", "Clean", "Dirty", "Stylized", "Sleepsizzing",
            "Sleek", "Nook", "Product", "Quality", "Good", "Fast", "Hyper", "Duty",
            "Important", "Bought", "Third", "Offensive", "NONE"
        ],
        "ProductOptions": [
            "Correct", "Unacceptable", "Waiting", "Overpriced", "Not Camera",
            "options", "choice", "selection", "alternative", "variant"
        ],
        "ProductWords": ["Wrong", "incorrect", "mistaken", "error", "mistake"],
        "Size": ["XL", "Adult", "Size", "Child", "large", "small", "medium", "fit"],
        "Damage": [
            "Fast delivery", "Battle", "Tasty", "Attack", "damage", "repair",
            "warranty", "Very poor", "broken", "scratched", "dented", "cracked",
            "torn", "ripped", "shattered"
        ]
    }
    
    # Preprocess keywords for better matching
    processed_classes = preprocess_keywords(base_classes)
    
    # Map topics to classes with weighted scoring
    topic_to_class = {}
    class_keywords = defaultdict(set)
    class_scores = defaultdict(int)  # Track class assignment frequency
    
    for topic_name, topic_words in topics.items():
        class_matches = {}
        
        for class_name, seed_words in processed_classes.items():
            # Count overlapping words with weighting
            overlap = set(topic_words).intersection(set(seed_words))
            score = len(overlap)
            
            # Store all matches with scores
            if score > 0:
                class_matches[class_name] = score
        
        # If no matches found, assign to "Other"
        if not class_matches:
            topic_to_class[topic_name] = "Other"
            class_keywords["Other"].update(topic_words)
        else:
            # Get top 2 matches if available
            sorted_matches = sorted(class_matches.items(), key=lambda x: x[1], reverse=True)
            top_matches = sorted_matches[:min(2, len(sorted_matches))]
            
            # Assign to top match
            best_match = top_matches[0][0]
            topic_to_class[topic_name] = best_match
            class_keywords[best_match].update(topic_words)
            class_scores[best_match] += 1
            
            # If there's a close second match, add those keywords too
            if len(top_matches) > 1 and top_matches[1][1] >= top_matches[0][1] * 0.7:
                second_match = top_matches[1][0]
                class_keywords[second_match].update(topic_words)
                class_scores[second_match] += 0.5  # Count as partial assignment
    
    # Log class assignment distribution
    logging.info("Class assignment distribution:")
    for class_name, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{class_name}: {score}")
    
    # Convert sets to lists
    for class_name in class_keywords:
        class_keywords[class_name] = list(class_keywords[class_name])
    
    return topic_to_class, dict(class_keywords)

def classify_review(review_text, review_title, class_keywords):
    """
    Classify a review into one or more classes based on expanded keywords
    with improved matching for better class distribution
    """
    stemmer = PorterStemmer()
    full_text = f"{review_title} {review_text}" if review_title else review_text
    full_text = full_text.lower()
    
    # Tokenize the text
    words = nltk.word_tokenize(full_text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    
    # Track class matches with scores
    class_scores = defaultdict(int)
    
    # Check for keywords in the text with improved matching
    for class_name, keywords in class_keywords.items():
        for keyword in keywords:
            # Check original keyword
            if keyword.lower() in full_text or re.search(r'\b' + re.escape(keyword.lower()) + r'\b', full_text):
                class_scores[class_name] += 2  # Higher score for exact matches
                continue
                
            # For multi-word keywords, check if most words are present
            if ' ' in keyword:
                keyword_words = keyword.lower().split()
                if len(keyword_words) > 1:
                    matches = sum(1 for word in keyword_words if word in full_text)
                    if matches >= len(keyword_words) * 0.7:  # 70% of words match
                        class_scores[class_name] += 1
                        continue
            
            # Check stemmed version
            stemmed_keyword = stemmer.stem(keyword.lower())
            if stemmed_keyword in stemmed_text:
                class_scores[class_name] += 1
    
    # Select classes with scores above threshold
    threshold = 1  # Minimum score to consider
    classes = [class_name for class_name, score in class_scores.items() if score >= threshold]
    
    # If no classes were found or only low-confidence matches, use a more targeted approach
    if not classes:
        # Check for specific indicators for each important class
        if any(word in full_text for word in ['damage', 'broken', 'crack', 'scratch', 'dent', 'repair']):
            classes.append('Damage')
        elif any(word in full_text for word in ['size', 'fit', 'large', 'small', 'tight', 'loose']):
            classes.append('Size')
        elif any(word in full_text for word in ['ship', 'deliver', 'arrival', 'package', 'box']):
            classes.append('Shipping')
        elif any(word in full_text for word in ['price', 'cost', 'expensive', 'cheap', 'worth']):
            classes.append('Priceperformance')
        elif any(word in full_text for word in ['service', 'support', 'help', 'contact', 'return']):
            classes.append('CustomerService')
    
    # If still no classes, default to 'Product'
    if not classes:
        classes = ['Product']
    elif 'Other' in classes and len(classes) > 1:
        classes.remove('Other')
    
    return classes

def convert_amazon_reviews_format(input_file, output_file):
    """Convert Amazon reviews to the format matching the Turkish dataset"""
    logging.info(f"Loading Amazon Reviews dataset from {input_file}...")
    
    # Read the dataset
    df = pd.read_csv(input_file)
    logging.info(f"Loaded {len(df)} reviews")
    
    # Extract topics from the dataset
    topics, _, _ = extract_topics_from_dataset(df)
    
    # Map topics to classes and get expanded keywords
    topic_to_class, class_keywords = map_topics_to_classes(topics)
    
    logging.info("Expanded class keywords:")
    for class_name, keywords in class_keywords.items():
        logging.info(f"{class_name}: {', '.join(keywords[:10])}...")
    
    # Create a new DataFrame with the required format
    new_df = pd.DataFrame()
    
    # Map review_body to Comment_en
    new_df['Comment_en'] = df['review_body']
    
    # Extract CommentClass_en based on review content with expanded keywords
    logging.info("Classifying reviews with expanded keywords...")
    new_df['CommentClass_en'] = df.apply(
        lambda row: classify_review(
            row['review_body'], 
            row.get('review_title', ''), 
            class_keywords
        ), 
        axis=1
    )
    
    # Use the index as Unnamed: 0
    new_df['Unnamed: 0'] = df.index
    
    # Log class distribution
    class_counts = defaultdict(int)
    for classes in new_df['CommentClass_en']:
        for cls in classes:
            class_counts[cls] += 1
    
    logging.info("Class distribution in classified reviews:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(new_df)) * 100
        logging.info(f"{class_name}: {count} ({percentage:.2f}%)")
    
    # Save to CSV
    logging.info(f"Saving converted format to {output_file}")
    new_df.to_csv(output_file, index=False)
    
    logging.info("Conversion completed successfully")
    return new_df

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "../datasets/exterim/english_amazonReviews.csv"  # Adjust path as needed
    output_file = "../datasets/interim/converted_amazonReviews.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert reviews format
    convert_amazon_reviews_format(input_file, output_file)

