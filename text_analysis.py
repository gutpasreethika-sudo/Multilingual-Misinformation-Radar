import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("Data/combined_news.csv")

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["title"])

print("Shape of Data:")
print(X.shape)

print("\nSample Features:")
print(vectorizer.get_feature_names_out()[:20])