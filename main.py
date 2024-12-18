import pandas as pd

matches = pd.read_csv("soccer-matches.csv", index_col=0)
print(matches.head())