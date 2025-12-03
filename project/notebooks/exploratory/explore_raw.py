import pandas as pd
from pathlib import Path

# Get the project root directory (3 levels up from this file)
# This file is in: project/notebooks/exploratory/explore_raw.py
# Project root is: project/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load from our project structure using absolute path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "consumer_behavior_dataset.csv"
print(f"Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# # Basic info
# print(df.info())
# print(df.describe())
# print(df.head())

# # Check for issues
# print(df.isnull().sum())  # Missing values
# print(df.duplicated().sum())  # Duplicates
# print(df["purchase_decision"].value_counts())  # Target distribution

# Conversion rate
conversion_rate = df["purchase_decision"].mean()
print(f"Conversion Rate: {conversion_rate:.2%}")

# Cart abandonment rate
cart_users = df[df["add_to_cart"] == 1]
abandonment_rate = cart_users["abandoned_cart"].mean()
print(f"Cart Abandonment Rate: {abandonment_rate:.2%}")

# Average order value (for purchasers)
df[df["purchase_decision"] == 1]["price"].describe()
# df[df["purchase_decision"] == 1]["price"].mean()
# df[df["purchase_decision"] == 1]["price"].median()


# Average discount applied
df[df["purchase_decision"] == 1]["discount_applied"].describe()
# df[df["purchase_decision"] == 1]["discount_applied"].mean()
# df[df["purchase_decision"] == 1]["discount_applied"].median()

# Average pages per session
df["pages_visited"].describe()

# Average time spent
df["time_spent"].describe()

# Weekend vs weekday purchases
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['purchase_date'].dt.dayofweek.describe()
df['is_weekend'] = df['purchase_date'].dt.dayofweek >= 5

df['purchase_decision'][df['is_weekend'] == 1].sum()
df['purchase_decision'][df['is_weekend'] == 0].sum()
weekend_purchases_percentile = df['purchase_decision'][df['is_weekend'] == 1].sum() / df['purchase_decision'].sum() * 100
weekday_purchases_percentile = df['purchase_decision'][df['is_weekend'] == 0].sum() / df['purchase_decision'].sum() * 100
print(f"Weekend purchases: {weekend_purchases_percentile:.2f}%")
print(f"Weekday purchases: {weekday_purchases_percentile:.2f}%")


# Engagement score
df['engagement_score'] = df['pages_visited'] * df['time_spent']
df['engagement_score'].describe()

# Price features
df['price_after_discount'] = df['price'] * (1 - df['discount_applied'] / 100)
df['discount_amount'] = df['price'] * df['discount_applied'] / 100

# Behavioral flags
df['high_engagement'] = (df['pages_visited'] > df['pages_visited'].median()).astype(int)
df['cart_converted'] = ((df['add_to_cart'] == 1) & (df['purchase_decision'] == 1)).astype(int)
df['high_engagement'].describe()
df['cart_converted'].describe()

df['purchase_decision'].value_counts(
    normalize=True
    )


# Missing values
print("Missing Values:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Unique values for categoricals
for col in ['category', 'gender', 'income_level', 'payment_method', 'location']:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())















