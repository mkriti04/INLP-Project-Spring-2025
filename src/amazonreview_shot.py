import pandas as pd

# 1) Load your full dataset
df = pd.read_csv('../datasets/interim/converted_amazonReviews.csv')

# 2) Take a reproducible random sample of 50k
df_sample = df.sample(n=50000, random_state=42)

# 3) Write it out
df_sample.to_csv('../datasets/interim/converted_amazonReviews_50k.csv', index=False)

print("Saved 50k‑row sample to converted_amazonReviews_50k.csv")
