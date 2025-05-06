import pandas as pd
import torch
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
from datasets import Dataset
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("./datasets/interim/translated_output_1.csv")  # Replace with your CSV filename

# Preprocess text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['Comment_en'].astype(str).apply(preprocess_text)
df['CommentClass_en'] = df['CommentClass_en'].apply(eval)
df['label'] = df['CommentClass_en'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")

# Generate positive pairs
input_examples = []
for label, group in df.groupby('label'):
    texts = group['cleaned_text'].tolist()
    # if len(texts) < 2:
    #     continue
    random.shuffle(texts)
    for i in range(0, len(texts) - 1, 2):
        input_examples.append(InputExample(texts=[texts[i], texts[i+1]]))

print(f"Created {len(input_examples)} training pairs")

# Load base SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune model
train_dataloader = DataLoader(input_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=20,
          warmup_steps=100,
          show_progress_bar=True)

# Save fine-tuned model
model.save("sbert_model_1")
torch.save(model, 'sbert_model_1.pt')
print("Saved fine-tuned model to 'sbert_model_1.pt'")

# Encode all comments using fine-tuned model
all_embeddings = model.encode(df['cleaned_text'].tolist(), convert_to_tensor=True)

# Bundle into dictionary and save as .pt
data_bundle = {
    'features': all_embeddings,
    'labels': df['CommentClass_en'].tolist(),
    'index': df['Unnamed: 0'].tolist()
}

torch.save(data_bundle, 'sbert_output_1.pt')
print("Saved embeddings to 'sbert_output_1.pt'")
