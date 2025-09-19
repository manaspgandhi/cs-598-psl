import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("bitcoin.csv")

# Parse and sort date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')

# ensure numeric
df['btc_market_price'] = pd.to_numeric(df['btc_market_price'], errors='coerce')

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

# Convert to numpy for plotting
x = df.index.to_numpy()
y = df['btc_market_price'].to_numpy(dtype=float)
plt.figure(figsize=(12,6))
plt.plot(x, y, label="BTC Market Price", color="blue")

# Find the first non-zero date
first_nonzero_date = df.loc[df['btc_market_price'] > 0].index.min()
first_nonzero_price = df.loc[first_nonzero_date, 'btc_market_price']

# Mark the point on the plot
plt.axvline(first_nonzero_date, color="red", linestyle="--", alpha=0.7, label=f"First non-zero price: {first_nonzero_date.date()}")
plt.scatter([first_nonzero_date], [first_nonzero_price], color="red", zorder=5)

# Labels and title
plt.title("Bitcoin Market Price Over Time")
plt.xlabel("Date")
plt.ylabel("BTC Market Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()
