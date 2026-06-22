import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("Data/combined_news.csv")
print(df.head())
print(df["label"].value_counts())
    
X = df["title"]

y = df["label"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
import joblib

joblib.dump(model, "Models/fake_news_model.pkl")
joblib.dump(vectorizer, "Models/vectorizer.pkl")

print("Model Saved Successfully!")
