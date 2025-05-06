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

# Example usage
if __name__ == "__main__":
    # Load dataset
    input_file = "../datasets/exterim/english_amazonReviews.csv"
    df = pd.read_csv(input_file)
    
    # Generate reasonable base classes using Mistral AI
    base_classes = generate_classes_with_mistral(df)
    
    # Print the generated base classes
    if base_classes:
        for class_name, keywords in base_classes.items():
            print(f"{class_name}: {', '.join(keywords[:10])}")