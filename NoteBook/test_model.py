import joblib

model = joblib.load("Models/fake_news_model.pkl")
vectorizer = joblib.load("Models/vectorizer.pkl")

text1 = "Free government money scheme available now"
text2 = "Government announces new education policy"

v1 = vectorizer.transform([text1])
v2 = vectorizer.transform([text2])

print("Fake test:", model.predict(v1))
print("Real test:", model.predict(v2))