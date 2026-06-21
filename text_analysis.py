import pandas as pd

df = pd.read_csv("Data/combined_news.csv")

print(df["title"].head())
print(df["text"].head())
