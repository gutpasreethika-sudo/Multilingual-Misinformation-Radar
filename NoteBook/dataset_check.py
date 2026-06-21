import pandas as pd

fake = pd.read_csv("Data/Fake.csv")
true = pd.read_csv("Data/True.csv")

print("Fake Dataset Shape:", fake.shape)
print("True Dataset Shape:", true.shape)

print("\nColumns:")
print(fake.columns)

print("\nFirst 5 Fake News Records:")
print(fake.head())