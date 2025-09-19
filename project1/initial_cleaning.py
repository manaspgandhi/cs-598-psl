import pandas as pd

df = pd.read_csv("bitcoin.csv")

# Parse and sort date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')

# Missing values
missing = df.isna().mean().sort_values(ascending=False) * 100
print("\%Percent of Missing Values by Column:\n", missing)

# Zero values
zeros = (df == 0).mean().sort_values(ascending=False) * 100
print("\n%Percent of Zero Values by Column:\n", zeros)

# Column summary
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "min": df.min(),
    "max": df.max()
})
print("\nColumn summary:\n", summary)