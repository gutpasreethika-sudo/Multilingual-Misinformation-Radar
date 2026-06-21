import pandas as pd
import re
# Load dataset
df = pd.read_csv("Data/combined_news.csv")

# Show original title
print("Original Title:")
print(df["title"].iloc[0])

# Convert to lowercase
df["title"] = df["title"].str.lower()

print("\nAfter Lowercase:")
print(df["title"].iloc[0])

# Remove punctuation
original_title = df["title"].iloc[0]

clean_title = re.sub(r'[^a-zA-Z\s]', '', original_title)

print("\nAfter Removing Punctuation:")
print(clean_title)
