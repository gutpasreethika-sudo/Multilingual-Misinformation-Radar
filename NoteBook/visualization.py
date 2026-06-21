import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/combined_news.csv")

counts = df["label"].value_counts()

counts.index = ["Real News", "Fake News"]

plt.figure(figsize=(6,4))
counts.plot(kind="bar")

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")

plt.show()
