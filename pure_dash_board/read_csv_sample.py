import pandas as pd

# Read the first few rows of the CSV file
df = pd.read_csv("2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv", nrows=5)

# Print the column names
print("Column names:")
print(df.columns.tolist())

# Print the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Print some basic statistics
print("\nBasic statistics:")
print(df.describe().transpose())

# Print data types
print("\nData types:")
print(df.dtypes)
