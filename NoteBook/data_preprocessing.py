import pandas as pd

# Load datasets
fake = pd.read_csv("Data/Fake.csv")
true = pd.read_csv("Data/True.csv")

# Add labels
fake["label"] = 1
true["label"] = 0

# Merge datasets
df = pd.concat([fake, true], ignore_index=True)

print("Total Records:", df.shape)

print("\nLabel Counts:")
print(df["label"].value_counts())

# Save combined dataset
df.to_csv("Data/combined_news.csv", index=False)

print("\nCombined dataset created successfully!")