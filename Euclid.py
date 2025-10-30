import pandas as pd

# 1. Read the CSV file into a pandas DataFrame
df = pd.read_csv("morphology_catalogue.csv")

# 2. Display the first few rows (optional)
print("First 5 rows:")
print(df.head())

# 3. Print a concise summary of the DataFrame
print("\nSummary info:")
print(df.info())
