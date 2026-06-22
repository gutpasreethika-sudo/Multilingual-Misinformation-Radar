import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("Data/combined_news.csv")
X = df["title"]
y = df["label"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X, y)
pickle.dump(model, open("Models/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("Models/vectorizer.pkl", "wb"))

print("Model Saved Successfully!")