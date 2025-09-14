import pandas as pd
import os

path = os.path.expanduser("~/Downloads/file_name.csv")
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumn Names and Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe(include='all'))
