import pandas as pd

df = pd.read_csv("data/enhanced_assignment3_dataset.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False).head(15))
print("\nSample rows:\n", df.head(5))