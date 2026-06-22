import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("Data/combined_news.csv")

stop_words = set(stopwords.words('english'))

text = df["title"].iloc[0].lower()

filtered_words = [word for word in text.split() if word not in stop_words]

print("Original:")
print(text)

print("\nAfter Stopword Removal:")
print(" ".join(filtered_words))