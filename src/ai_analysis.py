<<<<<<< HEAD
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import logging

# Load the Mistral model and tokenizer
def load_mistral_model():
    logging.info("Loading Mistral model...")
    model_name = "mistralai/Mistral-7B-v0.1"  # Replace with the correct model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def call_mistral_for_keywords(corpus, max_classes=10, model=None, tokenizer=None):
    """
    Use Mistral AI to analyze the dataset and generate reasonable keywords for classes.
    """
    logging.info("Using Mistral AI to analyze the dataset...")
    
    # Prepare the prompt for the model
    prompt = (
        "Analyze the following e-commerce reviews and generate up to "
        f"{max_classes} reasonable classes with associated keywords. "
        "Each class should represent a common theme in e-commerce reviews, "
        "such as product quality, pricing, delivery, or customer service. "
        "Provide the output in JSON format with class names as keys and keywords as values.\n\n"
        "Reviews:\n" + "\n".join(corpus[:100])  # Limit to 100 reviews for brevity
    )
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate the response
    outputs = model.generate(inputs.input_ids, max_length=1000, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logging.info("Mistral AI response received.")
    return result

def generate_classes_with_mistral(df, max_classes=10):
    """
    Generate reasonable classes for e-commerce reviews using Mistral AI.
    """
    # Combine title and review body if available
    if 'review_title' in df.columns:
        corpus = df['review_title'].fillna('') + ' ' + df['review_body'].fillna('')
    else:
        corpus = df['review_body'].fillna('')
    
    # Load the Mistral model and tokenizer
    model, tokenizer = load_mistral_model()
    
    # Call Mistral AI
    ai_response = call_mistral_for_keywords(corpus.tolist(), max_classes=max_classes, model=model, tokenizer=tokenizer)
    
    if ai_response:
        try:
            # Convert the JSON response to a dictionary
            base_classes = eval(ai_response)  # Use `json.loads` if the response is a valid JSON string
            logging.info("Generated reasonable base classes:")
            for class_name, keywords in base_classes.items():
                logging.info(f"{class_name}: {', '.join(keywords[:10])}...")
            return base_classes
        except Exception as e:
            logging.error(f"Error parsing AI response: {e}")
            return None
    else:
        logging.error("No response from Mistral AI.")
        return None

=======
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import logging
import os
import torch
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load a model and tokenizer using Hugging Face API key
def load_model(model_name="google/flan-t5-base"):
    """
    Load a model from Hugging Face using API key
    """
    logging.info(f"Loading model: {model_name}...")
    
    # Set your Hugging Face API key
    hf_api_key = "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"
    if not hf_api_key:
        logging.warning("HF_API_TOKEN not found in environment variables. You may face rate limits.")
    
    # Load tokenizer and model with API key
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
    
    # Explicitly set device_map to ensure consistent device usage
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, token=hf_api_key)
    
    return model, tokenizer

def filter_irrelevant_classes(classes):
    """
    Filter out irrelevant classes that don't make sense for e-commerce reviews
    and remove duplicate keywords within each class
    """
    # List of known irrelevant class names
    irrelevant_classes = [
        'alarm', 'correct', 'incorrect', 'general', 'sentiment', 'positive', 'negative', 
        'neutral', 'other', 'miscellaneous', 'misc', 'unknown'
    ]
    
    # List of known relevant class keywords (common e-commerce categories)
    relevant_class_keywords = [
        'product', 'quality', 'price', 'value', 'shipping', 'delivery', 'service', 
        'customer', 'packaging', 'feature', 'function', 'performance', 'design', 
        'durability', 'warranty', 'return', 'size', 'fit', 'color', 'material'
    ]
    
    filtered_classes = {}
    
    for class_name, keywords in classes.items():
        class_name_lower = class_name.lower()
        
        # Skip classes that match known irrelevant names
        if any(irrelevant in class_name_lower for irrelevant in irrelevant_classes):
            logging.info(f"Filtering out irrelevant class: {class_name}")
            continue
        
        # Keep classes that contain relevant keywords or have at least 3 relevant keywords
        relevant_keyword_count = sum(1 for keyword in relevant_class_keywords if keyword in class_name_lower)
        keyword_relevance = sum(1 for keyword in keywords if any(rel in keyword.lower() for rel in relevant_class_keywords))
        
        if relevant_keyword_count > 0 or keyword_relevance >= 3:
            # Remove duplicate keywords (case-insensitive)
            unique_keywords = []
            seen_keywords = set()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in seen_keywords:
                    unique_keywords.append(keyword)
                    seen_keywords.add(keyword_lower)
            
            filtered_classes[class_name] = unique_keywords
        else:
            logging.info(f"Filtering out potentially irrelevant class: {class_name}")
    
    # If we filtered everything, return at least some basic classes
    if not filtered_classes:
        logging.warning("All classes were filtered out, returning default classes")
        filtered_classes = {
            'ProductQuality': ['quality', 'durable', 'well-made', 'material', 'construction'],
            'Price': ['price', 'value', 'worth', 'expensive', 'cheap', 'cost'],
            'Shipping': ['shipping', 'delivery', 'arrived', 'package', 'box'],
            'CustomerService': ['service', 'support', 'customer', 'return', 'refund']
        }
    
    return filtered_classes

def compare_and_merge_classes(existing_classes, new_classes):
    """
    Compare existing classes with new ones and identify new classes
    Ensures no duplicate keywords within each class
    """
    merged_classes = existing_classes.copy()
    new_discovered_classes = {}
    
    for class_name, keywords in new_classes.items():
        # Check if this is a completely new class
        if class_name not in existing_classes:
            new_discovered_classes[class_name] = keywords
            merged_classes[class_name] = keywords
            logging.info(f"Discovered new class: {class_name}")
        else:
            # Check for new keywords in existing classes
            existing_keywords = set(k.lower() for k in existing_classes[class_name])
            new_keywords = [kw for kw in keywords if kw.lower() not in existing_keywords]
            
            if new_keywords:
                logging.info(f"Found new keywords for class '{class_name}': {new_keywords}")
                # Add new keywords while preserving case
                merged_classes[class_name] = existing_classes[class_name] + new_keywords
    
    return merged_classes, new_discovered_classes

def call_model_for_keywords(corpus, max_classes=10, model=None, tokenizer=None, model_name="google/flan-t5-base"):
    """
    Use a language model to analyze the dataset and generate reasonable keywords for classes.
    """
    logging.info(f"Using {model_name} to analyze the dataset...")
    
    # Take a small sample of reviews to stay within token limits
    sample_reviews = corpus[:5]  # Just use 5 reviews for the prompt
    
    # Create a more specific prompt that emphasizes e-commerce relevance and no duplicate keywords
    prompt = (
        f"You are analyzing e-commerce product reviews. Generate exactly {max_classes} relevant e-commerce review "
        f"classification categories that customers commonly discuss in their feedback. "
        f"Focus ONLY on standard e-commerce categories like Product Quality, Price, Shipping, Customer Service, etc. "
        f"DO NOT include irrelevant categories like 'Alarm', 'Correct', or general sentiment classes. "
        f"For each category, provide 5-10 UNIQUE keywords that would help identify reviews belonging to that category. "
        f"DO NOT repeat keywords within a class or across classes. "
        f"Format your response as a Python dictionary with class names as keys and keyword lists as values. "
        f"Example: {{'ProductQuality': ['quality', 'durable', 'well-made'], 'Price': ['expensive', 'cheap', 'worth']}}. "
        f"Here are some sample reviews to analyze: " + " ".join(sample_reviews)
    )
    
    # Check token length before processing
    tokens = tokenizer(prompt, return_tensors="pt")
    if tokens.input_ids.shape[1] > 512:
        logging.warning(f"Input too long: {tokens.input_ids.shape[1]} tokens. Truncating...")
        # Keep the instruction part and truncate only the reviews
        instruction = (
            f"You are analyzing e-commerce product reviews. Generate exactly {max_classes} relevant e-commerce review "
            f"classification categories that customers commonly discuss in their feedback. "
            f"Focus ONLY on standard e-commerce categories like Product Quality, Price, Shipping, Customer Service, etc. "
            f"DO NOT include irrelevant categories like 'Alarm', 'Correct', or general sentiment classes. "
            f"For each category, provide 5-10 UNIQUE keywords that would help identify reviews belonging to that category. "
            f"DO NOT repeat keywords within a class or across classes. "
            f"Format your response as a Python dictionary with class names as keys and keyword lists as values. "
            f"Example: {{'ProductQuality': ['quality', 'durable', 'well-made'], 'Price': ['expensive', 'cheap', 'worth']}}. "
            f"Here are some sample reviews to analyze: "
        )
        remaining_tokens = 512 - len(tokenizer(instruction, return_tensors="pt").input_ids[0])
        truncated_reviews = tokenizer.decode(tokens.input_ids[0][-remaining_tokens:], skip_special_tokens=True)
        prompt = instruction + truncated_reviews
        tokens = tokenizer(prompt, return_tensors="pt")
    
    # Rest of the function remains the same
    device = next(model.parameters()).device
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=1000, 
        temperature=0.7,
        do_sample=True
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Model response: {result}")
    
    return result



def load_existing_classes():
    """
    Load existing classes from a file if available
    """
    results_dir = "../results/ai_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    existing_classes_file = os.path.join(results_dir, "existing_classes.json")
    if os.path.exists(existing_classes_file):
        try:
            with open(existing_classes_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading existing classes: {e}")
    
    # Default classes if file doesn't exist
    return {
        'Product': ['quality', 'design', 'material', 'product', 'item'],
        'Price': ['price', 'value', 'worth', 'expensive', 'cheap'],
        'Shipping': ['shipping', 'delivery', 'arrived', 'package'],
        'CustomerService': ['service', 'support', 'customer', 'return']
    }

def save_classes(classes, filename):
    """
    Save classes to a JSON file
    """
    results_dir = "../results/ai_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(classes, f, indent=2)
    
    logging.info(f"Saved classes to {filepath}")
    return filepath



def parse_model_response(response_text):
    """
    Parse the model's response and extract class information even if it's not in proper JSON format.
    Returns a dictionary of classes and their keywords.
    """
    classes = {}
    
    # Try JSON parsing first
    try:
        import re
        import json
        
        # Look for JSON-like structure in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Try to parse the JSON
            try:
                classes = json.loads(json_str)
                return classes
            except json.JSONDecodeError:
                logging.warning(f"Could not parse JSON: {json_str}")
        else:
            logging.warning("No JSON structure found in the response")
    except Exception as e:
        logging.warning(f"Error in JSON parsing: {e}")
    
    # If JSON parsing fails, try regex-based extraction
    try:
        # Pattern to match class names and their keywords
        # Looks for patterns like "Class: [keyword1, keyword2, ...]" or "Class: keyword1, keyword2, ..."
        class_pattern = r"['\"]([\w\s]+)['\"]:\s*\[(.*?)\]|['\"]([\w\s]+)['\"]:\s*\[(.*?)\]|(\w+):\s*\[(.*?)\]|(\w+):\s*([^\[\]]+?)(?:,\s*['\"]|\s*\})"
        matches = re.findall(class_pattern, response_text, re.DOTALL)
        
        for match in matches:
            # Extract class name (could be in different capture groups)
            class_name = next((m for m in match[:7:2] if m), None)
            
            # Extract keywords (could be in different capture groups)
            keywords_str = next((m for m in match[1:8:2] if m), None)
            
            if class_name and keywords_str:
                # Clean up class name
                class_name = class_name.strip()
                
                # Extract keywords from the string
                keywords = []
                keyword_pattern = r"['\"]([\w\s]+)['\"]|(\w+)"
                keyword_matches = re.findall(keyword_pattern, keywords_str)
                
                for keyword_match in keyword_matches:
                    keyword = next((k for k in keyword_match if k), None)
                    if keyword:
                        keywords.append(keyword.strip())
                
                if keywords:
                    classes[class_name] = keywords
    except Exception as e:
        logging.warning(f"Error in regex parsing: {e}")
    
    # If all parsing fails, try a simple line-by-line approach
    if not classes:
        try:
            lines = response_text.split('\n')
            current_class = None
            
            for line in lines:
                line = line.strip()
                
                # Check if this line defines a class
                class_match = re.match(r"['\"]?([\w\s]+)['\"]?\s*:", line)
                if class_match:
                    current_class = class_match.group(1).strip()
                    classes[current_class] = []
                    
                    # Extract keywords from the same line
                    keywords_part = line.split(':', 1)[1].strip()
                    if keywords_part:
                        # Extract keywords
                        keyword_matches = re.findall(r"['\"]([\w\s]+)['\"]|(\w+)", keywords_part)
                        for keyword_match in keyword_matches:
                            keyword = next((k for k in keyword_match if k), None)
                            if keyword:
                                classes[current_class].append(keyword.strip())
                
                # If we're in a class and this line has keywords
                elif current_class and line and not line.startswith('{') and not line.startswith('}'):
                    # Extract keywords
                    keyword_matches = re.findall(r"['\"]([\w\s]+)['\"]|(\w+)", line)
                    for keyword_match in keyword_matches:
                        keyword = next((k for k in keyword_match if k), None)
                        if keyword:
                            classes[current_class].append(keyword.strip())
        except Exception as e:
            logging.warning(f"Error in line-by-line parsing: {e}")
    
    # If we still couldn't parse anything, return a default structure
    if not classes:
        logging.warning("Could not parse response, using default classes")
        classes = {
            'Product': ['quality', 'design', 'material', 'product', 'item'],
            'Price': ['price', 'value', 'worth', 'expensive', 'cheap'],
            'Shipping': ['shipping', 'delivery', 'arrived', 'package'],
            'CustomerService': ['service', 'support', 'customer', 'return']
        }
    
    return classes

def generate_classes_with_model(df, max_classes=10, model_name="google/flan-t5-base", save_interval=5, sample_limit=2000):
    """
    Generate reasonable classes for e-commerce reviews using a language model.
    Periodically saves results and identifies new classes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing reviews
    max_classes : int
        Maximum number of classes to generate
    model_name : str
        Name of the Hugging Face model to use
    save_interval : int
        How often to save results (in terms of batches)
    sample_limit : int
        Number of samples to process from the dataset
    """
    # Create results directory
    results_dir = "../results/ai_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load existing classes
    existing_classes = load_existing_classes()
    logging.info(f"Loaded existing classes: {list(existing_classes.keys())}")
    
    # Limit to first sample_limit entries
    df_limited = df.head(sample_limit)
    logging.info(f"Limited dataset to first {sample_limit} entries out of {len(df)}")
    
    # Combine title and review body if available
    if 'review_title' in df_limited.columns:
        corpus = df_limited['review_title'].fillna('') + ' ' + df_limited['review_body'].fillna('')
    else:
        corpus = df_limited['review_body'].fillna('')
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Process in batches to save periodically
    batch_size = min(save_interval, len(corpus))
    num_batches = (len(corpus) + batch_size - 1) // batch_size
    
    all_discovered_classes = {}
    merged_classes = existing_classes.copy()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(corpus))
        batch_corpus = corpus[start_idx:end_idx].tolist()
        
        logging.info(f"Processing batch {batch_idx+1}/{num_batches} (samples {start_idx}-{end_idx})")
        
        # Call the model
        ai_response = call_model_for_keywords(batch_corpus, max_classes=max_classes, 
                                             model=model, tokenizer=tokenizer, model_name=model_name)
        
        if ai_response:
            try:
                # Try to extract JSON from the response
                import re
                import json
                
                # Look for JSON-like structure in the response
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Try to parse the JSON
                    try:
                        batch_classes = json.loads(json_str)
                        
                        # Filter out irrelevant classes
                        filtered_classes = filter_irrelevant_classes(batch_classes)
                        
                        # Compare with existing classes and identify new ones
                        merged_classes, new_classes = compare_and_merge_classes(merged_classes, filtered_classes)
                        
                        # Save the new classes discovered in this batch
                        if new_classes:
                            all_discovered_classes.update(new_classes)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_classes(new_classes, f"new_classes_{timestamp}.json")
                        
                        # Save the merged classes
                        save_classes(merged_classes, "existing_classes.json")
                        
                    except json.JSONDecodeError:
                        logging.error(f"Could not parse JSON: {json_str}")
                else:
                    logging.warning("No JSON structure found in the response")
                    # Try to parse the response using a custom parser
                    batch_classes = parse_model_response(ai_response)
                    
                    # Filter out irrelevant classes
                    filtered_classes = filter_irrelevant_classes(batch_classes)
                    
                    # Compare with existing classes and identify new ones
                    merged_classes, new_classes = compare_and_merge_classes(merged_classes, filtered_classes)
                    
                    # Save the new classes discovered in this batch
                    if new_classes:
                        all_discovered_classes.update(new_classes)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_classes(new_classes, f"new_classes_{timestamp}.json")
                    
                    # Save the merged classes
                    save_classes(merged_classes, "existing_classes.json")
                
            except Exception as e:
                logging.error(f"Error parsing AI response: {e}")
        else:
            logging.error(f"No response from {model_name}.")
    
    # Save final results
    if all_discovered_classes:
        save_classes(all_discovered_classes, "all_discovered_classes.json")
    
    return merged_classes



# Example usage
>>>>>>> c9911334d70faef7ab964419f234f9311fcf8d1d
# Example usage
if __name__ == "__main__":
    # Load dataset
    input_file = "../datasets/exterim/english_amazonReviews.csv"
    df = pd.read_csv(input_file)
    
<<<<<<< HEAD
    # Generate reasonable base classes using Mistral AI
    base_classes = generate_classes_with_mistral(df)
=======
    # Set your Hugging Face API key as an environment variable
    # os.environ["HF_API_TOKEN"] = "your_api_key_here"  # Uncomment and replace with your key
    
    # Choose a different model
    model_name = "google/flan-t5-base"  # You can replace with any model you prefer
    
    # Generate reasonable base classes using the selected model, processing only 2000 samples
    base_classes = generate_classes_with_model(df, model_name=model_name, sample_limit=4000)
>>>>>>> c9911334d70faef7ab964419f234f9311fcf8d1d
    
    # Print the generated base classes
    if base_classes:
        for class_name, keywords in base_classes.items():
<<<<<<< HEAD
            print(f"{class_name}: {', '.join(keywords[:10])}")
=======
            print(f"{class_name}: {', '.join(keywords[:10])}")

>>>>>>> c9911334d70faef7ab964419f234f9311fcf8d1d
