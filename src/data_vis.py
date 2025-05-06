import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import nltk
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

# Create figures folder
os.makedirs("../figures", exist_ok=True)

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset
df = pd.read_csv("../datasets/interim/translated_output_1.csv")
df['CommentClass_en'] = df['CommentClass_en'].apply(ast.literal_eval)

# 1. Label Distribution (Improved)
all_labels = [label for labels in df['CommentClass_en'] for label in labels]
label_counts = Counter(all_labels)

# Sort labels by frequency
labels_sorted, counts_sorted = zip(*label_counts.most_common())

plt.figure(figsize=(10, 5))  # Increased size
plt.bar(labels_sorted, counts_sorted, color='skyblue')
plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')  # Rotate and align
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("../figures/label_distribution.png")
plt.close()


# 2. Comment Length Distribution
df['comment_length'] = df['Comment_en'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(6, 4))
plt.hist(df['comment_length'], bins=10, color='salmon', edgecolor='black')
plt.title("Comment Length Distribution (in words)")
plt.xlabel("Number of Words")
plt.ylabel("Number of Comments")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("../figures/comment_length_distribution.png")
plt.close()

# 3. Most Frequent Words
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

df['tokens'] = df['Comment_en'].apply(preprocess)
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words).most_common(15)

words, freqs = zip(*word_freq)
plt.figure(figsize=(8, 4))
plt.bar(words, freqs, color='mediumpurple')
plt.xticks(rotation=45)
plt.title("Top 15 Most Common Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../figures/most_common_words.png")
plt.close()

# 4. Label Co-occurrence Heatmap
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(df['CommentClass_en'])

co_matrix = pd.DataFrame(label_matrix, columns=mlb.classes_).T.dot(label_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(co_matrix, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Label Co-occurrence Heatmap")
plt.tight_layout()
plt.savefig("../figures/label_cooccurrence_heatmap.png")
plt.close()

print("EDA visualizations saved in ../figures/")
