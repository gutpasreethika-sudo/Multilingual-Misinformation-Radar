import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv("Data/combined_news.csv")

text = df["title"].iloc[0].lower()

words = text.split()

filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]

print("Original:")
print(text)

print("\nAfter Stop Word Removal:")
print(" ".join(filtered_words))
