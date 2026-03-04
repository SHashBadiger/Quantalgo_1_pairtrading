import pandas as pd

# Load the files
df1 = pd.read_csv('2024-2025.csv')
df2 = pd.read_csv('2025-2026.csv')

# Create sets of tuples for the pairs (Stock 1, Stock 2)
pairs1 = set(zip(df1['Stock 1'], df1['Stock 2']))
pairs2 = set(zip(df2['Stock 1'], df2['Stock 2']))

# Find exact matches
repeating_pairs = pairs1.intersection(pairs2)
print("Repeating Pairs:", repeating_pairs)

# To find common individual stocks
'''stocks1 = set(df1['Stock 1']).union(df1['Stock 2'])
stocks2 = set(df2['Stock 1']).union(df2['Stock 2'])
common_stocks = stocks1.intersection(stocks2)
print("Common Individual Stocks:", common_stocks)'''