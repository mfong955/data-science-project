import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Get the project root directory (3 levels up from this file)
# This file is in: project/notebooks/exploratory/explore_raw.py
# Project root is: project/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load from our project structure using absolute path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "consumer_behavior_dataset.csv"
print(f"Loading data from: {DATA_PATH}")

# Define path for saving figures (as per 03_ANALYSIS_PLAN.md line 473)
FIGURES_PATH = PROJECT_ROOT / "visualizations" / "figures"
# Ensure the figures directory exists
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_PATH}")

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
df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["purchase_date"].dt.dayofweek.describe()
df["is_weekend"] = df["purchase_date"].dt.dayofweek >= 5

df["purchase_decision"][df["is_weekend"] == 1].sum()
df["purchase_decision"][df["is_weekend"] == 0].sum()
weekend_purchases_percentile = (
    df["purchase_decision"][df["is_weekend"] == 1].sum()
    / df["purchase_decision"].sum()
    * 100
)
weekday_purchases_percentile = (
    df["purchase_decision"][df["is_weekend"] == 0].sum()
    / df["purchase_decision"].sum()
    * 100
)
print(f"Weekend purchases: {weekend_purchases_percentile:.2f}%")
print(f"Weekday purchases: {weekday_purchases_percentile:.2f}%")


# Engagement score
df["engagement_score"] = df["pages_visited"] * df["time_spent"]
df["engagement_score"].describe()

# Price features
df["price_after_discount"] = df["price"] * (1 - df["discount_applied"] / 100)
df["discount_amount"] = df["price"] * df["discount_applied"] / 100

# Behavioral flags
df["high_engagement"] = (df["pages_visited"] > df["pages_visited"].median()).astype(int)
df["cart_converted"] = (
    (df["add_to_cart"] == 1) & (df["purchase_decision"] == 1)
).astype(int)
df["high_engagement"].describe()
df["cart_converted"].describe()

df["purchase_decision"].value_counts(normalize=True)


# Missing values
print("Missing Values:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Unique values for categoricals
for col in ["category", "gender", "income_level", "payment_method", "location"]:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())

#### Objectives
# - Understand data structure and quality
# - Define key product metrics
# - Identify initial patterns and relationships
# - Create baseline for comparison

#### Key Questions
# 1. What's the overall conversion rate?
# 2. How is the target variable distributed?
# 3. Are there missing values or data quality issues?
# 4. What are the distributions of key features?
# 5. How do features correlate with purchase decisions?

#### Analyses to Perform

# **Data Quality Assessment**
# Load from project structure
df = pd.read_csv("../../data/raw/consumer_behavior_dataset.csv")

# Missing values
df.isnull().sum()

# Duplicates
df.duplicated().sum()

# Outliers
df.sort_index(axis=0).describe()
# Need to plot histograms/boxplots for better outlier detection

# Data type verification
df.dtypes

# Logical consistency checks

# Check for impossible values (e.g., negative prices)
df["price"][df["price"] < 0]

# Verify date ranges
df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["purchase_date"].min(), df["purchase_date"].max()

# Check for inconsistent categories
df["category"].unique()

# Validate relationships between features
df["add_to_cart"][df["purchase_decision"] == 1].value_counts()
sum(df["add_to_cart"]) - sum(df["abandoned_cart"])
sum(df["purchase_decision"])

sum(df["abandoned_cart"])
sum(df["add_to_cart"][df["abandoned_cart"] == 1])
sum(
    df["add_to_cart"][df["abandoned_cart"] == 0]
)  # matches sum(df['purchase_decision'])


# **Univariate Analysis**
# Distribution of numeric features (histograms, box plots)
# Frequency of categorical features (bar charts)
# Summary statistics for all features
# Each purhcase has an associated day. I will convert the purchase_date to week (ind)
# Convert purchase_date to purchase_week (1, 2, 3, etc.)
min_week = df["purchase_date"].dt.isocalendar().week.min()
df["purchase_week"] = df["purchase_date"].dt.isocalendar().week - min_week + 1
df["weekly_purchases"] = (
    df.groupby(pd.Grouper(key="purchase_date", freq="W"))["purchase_date"]
    .transform("size")
    .astype("int64")
)


cols = [
    "category",
    "price",
    "discount_applied",
    "payment_method",
    # "purchase_date",
    "purchase_week",
    "pages_visited",
    "time_spent",
    "add_to_cart",
    "abandoned_cart",
    "rating",
    "sentiment_score",
    "age",
    "gender",
    "income_level",
    "purchase_decision",
]
for col in cols:  # Only loop over non user/product-ids
    plt.figure(figsize=(8, 5))  # Create a new figure for each plot
    print(df[col].describe())
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].plot(kind="hist", title=f"Histogram of {col}")
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col].value_counts().sort_index().plot(
            kind="bar", title=f"Frequency of {col} over time"
        )
    elif pd.api.types.is_object_dtype(df[col]):
        df[col].value_counts().plot(kind="bar", title=f"Value counts of {col}")

    # plt.savefig() - Saves the current figure to a file (first use)
    # Parameters:
    #   fname - File path (can be string or Path object)
    #   dpi - Resolution in dots per inch (higher = better quality)
    #   bbox_inches='tight' - Removes extra whitespace around the figure
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f"univariate_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()  # Close figure to free memory


# **Bivariate Analysis**
# Conversion rate by category (product, demographics, etc.)
categories = ["category", "rating", "purchase_week", "gender", "age"]

for cats in categories:
    conversion_by_category = df.groupby(cats).agg(
        {"purchase_decision": ["sum", "count"]}
    )
    conversion_by_category.columns = ["Purchases", "Total_Records"]
    conversion_by_category.plot(kind="bar")
    plt.title(f"Purchases and Total Records by {cats}")
    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / f"bivariate_{cats}_counts.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    df.groupby(cats)["purchase_decision"].mean().plot(
        kind="bar", title=f"Conversion Rate by {cats}"
    )
    plt.ylabel("Conversion Rate")
    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / f"bivariate_{cats}_conversion.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

# **Bivariate Analysis**
# Let's explore relationships between different features and purchase decisions

# ============================================================================
# PRICE VS CONVERSION ANALYSIS
# ============================================================================
print("\n=== PRICE VS CONVERSION ANALYSIS ===")

# pd.cut() - Bins continuous data into discrete intervals (first use)
# Creates 10 equal-width bins from min to max price values
# Returns a Categorical Series with interval labels like (0, 100], (100, 200], etc.
price_bins = pd.cut(df["price"], bins=10)

# .groupby() - Groups DataFrame by specified column(s) for aggregation (used earlier)
# .agg() - Applies multiple aggregation functions at once (first use)
# Here we calculate both mean (conversion rate) and count (sample size) per price bin
price_conversion = df.groupby(price_bins)["purchase_decision"].agg(["mean", "count"])
price_conversion.columns = ["Conversion_Rate", "Total_Sessions"]
print("Conversion rate by price range:")
print(price_conversion)

# Visualize price vs conversion
# plt.figure() - Creates a new figure with specified size in inches (used earlier)
# figsize=(width, height) controls the plot dimensions
plt.figure(figsize=(10, 6))

# .plot(kind="bar") - Creates a bar chart from Series/DataFrame (used earlier)
# color parameter sets the bar fill color
price_conversion["Conversion_Rate"].plot(kind="bar", color="skyblue")

# plt.title() - Sets the chart title (used earlier)
plt.title("Conversion Rate by Price Range")

# plt.xlabel() / plt.ylabel() - Set axis labels (first use)
plt.xlabel("Price Range")
plt.ylabel("Conversion Rate")

# plt.xticks(rotation=45) - Rotates x-axis tick labels by 45 degrees (first use)
# Useful when labels are long and would overlap horizontally
plt.xticks(rotation=45)

# plt.tight_layout() - Automatically adjusts subplot params for better fit (first use)
# Prevents labels from being cut off at figure edges
plt.tight_layout()

# plt.savefig() - Saves the current figure to a file
# dpi=150 sets resolution, bbox_inches='tight' removes extra whitespace
plt.savefig(FIGURES_PATH / "price_vs_conversion.png", dpi=150, bbox_inches="tight")

# plt.show() - Displays the figure (optional when saving)
# Renders the plot and clears the current figure
plt.show()
plt.close()

# ============================================================================
# SESSION DURATION VS CONVERSION ANALYSIS
# ============================================================================
print("\n=== SESSION DURATION VS CONVERSION ANALYSIS ===")

# Same binning approach as price - divides time_spent into 10 equal-width intervals
time_bins = pd.cut(df["time_spent"], bins=10)
time_conversion = df.groupby(time_bins)["purchase_decision"].agg(["mean", "count"])
time_conversion.columns = ["Conversion_Rate", "Total_Sessions"]
print("Conversion rate by session duration:")
print(time_conversion)

# Visualize session duration vs conversion (same pattern as price visualization)
plt.figure(figsize=(10, 6))
time_conversion["Conversion_Rate"].plot(kind="bar", color="lightgreen")
plt.title("Conversion Rate by Session Duration")
plt.xlabel("Time Spent Range (minutes)")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "session_duration_vs_conversion.png", dpi=150, bbox_inches="tight"
)
plt.show()
plt.close()

# ============================================================================
# CORRELATION MATRIX
# ============================================================================
print("\n=== CORRELATION MATRIX ===")

# .select_dtypes() - Filters columns by data type (first use)
# include=[np.number] selects all numeric columns (int, float)
# Returns a DataFrame with only the matching columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# .corr() - Computes pairwise Pearson correlation coefficients (first use)
# Returns a square DataFrame where each cell [i,j] is correlation between columns i and j
# Values range from -1 (perfect negative) to +1 (perfect positive), 0 = no correlation
correlation_matrix = df[numeric_cols].corr()

# Display correlation with purchase_decision
print("Correlation with purchase_decision (target variable):")

# .sort_values() - Sorts Series by values (used earlier)
# ascending=False puts highest correlations first
purchase_correlations = correlation_matrix["purchase_decision"].sort_values(
    ascending=False
)
print(purchase_correlations)

# Visualize correlation matrix as heatmap
plt.figure(figsize=(12, 10))

# sns.heatmap() - Creates a color-coded matrix visualization (first use)
# Parameters:
#   annot=True - Display correlation values in each cell
#   cmap="coolwarm" - Color palette (blue for negative, red for positive)
#   center=0 - Center the colormap at 0 (neutral correlation)
#   fmt=".2f" - Format annotations to 2 decimal places
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Matrix of All Numeric Features")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================================
# KEY METRICS CALCULATION
# ============================================================================
print("\n=== KEY METRICS CALCULATION ===")

# 1. Overall conversion rate
# .mean() on a binary column (0/1) gives the proportion of 1s (conversion rate)
overall_conversion = df["purchase_decision"].mean()
print(f"1. Overall Conversion Rate: {overall_conversion:.2%}")

# 2. Cart abandonment rate
# Filter to users who added to cart, then calculate abandonment proportion
cart_abandonment = df[df["add_to_cart"] == 1]["abandoned_cart"].mean()
print(f"2. Cart Abandonment Rate: {cart_abandonment:.2%}")

# 3. Average Order Value (AOV) - only for actual purchases
# Filter to purchasers only, then calculate mean price
aov = df[df["purchase_decision"] == 1]["price"].mean()
print(f"3. Average Order Value (AOV): ${aov:.2f}")

# 4. Average session duration
avg_session_duration = df["time_spent"].mean()
print(f"4. Average Session Duration: {avg_session_duration:.2f} seconds")

# 5. Average pages per session
avg_pages_per_session = df["pages_visited"].mean()
print(f"5. Average Pages per Session: {avg_pages_per_session:.2f}")

# 6. Bounce rate (1-page sessions)
# Boolean comparison returns True/False, .mean() gives proportion of True values
bounce_rate = (df["pages_visited"] == 1).mean()
print(f"6. Bounce Rate (1-page sessions): {bounce_rate:.2%}")

# ============================================================================
# ADDITIONAL INSIGHTS
# ============================================================================
print("\n=== ADDITIONAL INSIGHTS ===")

# len(df) - Returns number of rows in DataFrame (used earlier)
print(f"Total sessions: {len(df)}")

# .sum() on binary column counts the number of 1s (purchases)
print(f"Total purchases: {df['purchase_decision'].sum()}")

# Chain filter and sum to get total revenue from purchasers
print(f"Total revenue: ${df[df['purchase_decision'] == 1]['price'].sum():.2f}")
print(
    f"Average discount applied: {df[df['purchase_decision'] == 1]['discount_applied'].mean():.2f}%"
)
