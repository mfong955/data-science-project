# Dataset Information

## 📦 AI-Driven Consumer Behavior Dataset

**Source:** Kaggle (2025)  
**Creator:** ziya07  
**License:** CC0: Public Domain  
**Size:** 1.17 MB  
**Format:** CSV  
**Rows:** ~10,000+ customer sessions

---

## 📋 Dataset Overview

This dataset captures comprehensive e-commerce user behavior including:
- Purchase decisions and transactions
- Clickstream/browsing behavior
- Customer reviews and sentiment
- Demographic information
- Product and pricing details

**Perfect for demonstrating:**
- Product analytics workflows
- Customer segmentation
- Predictive modeling
- A/B test analysis
- Business intelligence

---

## 🗂️ Dataset Structure

### Key Feature Categories

#### 1. **Purchase Data**
Information about transactions and buying behavior

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `purchase_decision` | Binary | **TARGET VARIABLE** - Did customer purchase? (1=Yes, 0=No) | 0, 1 |
| `product_category` | Categorical | Type of product viewed/purchased | Electronics, Clothing, Home & Garden |
| `price` | Numeric | Product price in dollars | 15.99, 299.00, 1249.99 |
| `discount` | Numeric | Discount percentage applied | 0, 10, 15, 20, 25 |
| `payment_method` | Categorical | Method of payment used | Credit Card, PayPal, Debit Card |

#### 2. **Clickstream Data**
User browsing and engagement patterns

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `pages_visited` | Integer | Number of pages viewed in session | 1, 5, 12, 20 |
| `session_duration` | Numeric | Time spent on site (minutes) | 2.5, 7.1, 15.8 |
| `cart_added` | Binary | Did user add items to cart? | 0, 1 |
| `cart_abandoned` | Binary | Did user abandon cart? | 0, 1 |
| `time_of_day` | Categorical | When session occurred | Morning, Afternoon, Evening, Night |
| `day_of_week` | Categorical | Day of session | Monday, Tuesday, etc. |

#### 3. **Customer Reviews & Sentiment**
Feedback and opinion data

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `rating` | Integer | Customer rating (1-5 stars) | 1, 2, 3, 4, 5 |
| `review_text` | Text | Written review content | "Great product!", "Not as described" |
| `sentiment_score` | Numeric | Sentiment analysis score (-1 to 1) | -0.5 (negative), 0.8 (positive) |

#### 4. **Demographics**
Customer characteristics

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `age` | Integer | Customer age | 18, 25, 35, 42, 65 |
| `gender` | Categorical | Customer gender | Male, Female, Other |
| `location` | Categorical | Geographic location | USA, UK, Canada, Germany |
| `income_level` | Categorical | Income bracket | Low, Medium, High |

---

## 📊 Expected Data Characteristics

### Target Variable Distribution
- **Conversion Rate:** Likely 10-20% (typical for e-commerce)
- **Class Imbalance:** More non-purchases than purchases
- **Implication:** Need to handle imbalanced classes in modeling

### Numeric Features
- **Price:** Wide range ($10 - $5000+), right-skewed distribution
- **Session Duration:** Log-normal distribution (most short, some very long)
- **Pages Visited:** Poisson-like distribution
- **Sentiment Score:** Centered around positive (0.3-0.7)

### Categorical Features
- **Product Category:** 5-10 categories, some more popular
- **Demographics:** Representative distribution across age/gender/location
- **Time Features:** Patterns by day/time (weekday vs weekend)

---

## 🎯 Key Business Questions This Data Can Answer

### Product Analytics
1. What's our overall conversion rate?
2. Which product categories perform best?
3. How does cart abandonment vary by segment?
4. What's the average order value (AOV)?

### User Behavior
5. How many pages do users typically visit before purchasing?
6. What's the optimal session duration for conversion?
7. When do users shop most (day/time patterns)?
8. What's the relationship between engagement and purchase?

### Customer Segmentation
9. Can we identify distinct user personas?
10. Which segments have highest lifetime value potential?
11. How do demographics affect behavior?
12. What are characteristics of power users?

### Pricing & Promotions
13. How do discounts impact conversion?
14. What's the price elasticity of demand?
15. Which discount levels are most effective?
16. Do discounts cannibalize full-price sales?

### Predictive Analytics
17. Can we predict purchase likelihood?
18. What features best predict conversion?
19. Can we identify cart abandonment risk early?
20. What intervention strategies might work?

---

## 🔍 Data Quality Checks to Perform

### Initial Exploration
```python
# Load and inspect
df = pd.read_csv('data/consumer_behavior_dataset.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head())

# Check for issues
print(df.isnull().sum())  # Missing values
print(df.duplicated().sum())  # Duplicates
print(df['purchase_decision'].value_counts())  # Target distribution
```

### Expected Checks
- [ ] **Missing values:** Check each column
- [ ] **Duplicates:** Identify and handle
- [ ] **Outliers:** Extreme prices or session durations
- [ ] **Data types:** Ensure correct types
- [ ] **Value ranges:** Logical bounds (age > 0, rating 1-5)
- [ ] **Consistency:** Logical relationships (cart_added if purchase_decision=1)

---

## 🛠️ Feature Engineering Opportunities

### Behavioral Features
```python
# Engagement score
df['engagement_score'] = df['pages_visited'] * df['session_duration']

# Time per page
df['time_per_page'] = df['session_duration'] / df['pages_visited']

# Discount sensitivity
df['discounted'] = (df['discount'] > 0).astype(int)

# High value purchase
df['high_value'] = (df['price'] > df['price'].median()).astype(int)
```

### Temporal Features
```python
# Weekend flag
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

# Peak hours
df['is_peak_hours'] = df['time_of_day'].isin(['Afternoon', 'Evening'])
```

### Interaction Features
```python
# Price-income fit
df['price_income_ratio'] = df['price'] / df['income_level'].map({
    'Low': 30000, 'Medium': 60000, 'High': 100000
})

# Engaged cart abandoner (high engagement but no purchase)
df['engaged_abandoner'] = (
    (df['pages_visited'] > 5) & 
    (df['cart_added'] == 1) & 
    (df['purchase_decision'] == 0)
)
```

---

## 📈 Expected Insights

Based on typical e-commerce patterns, expect to find:

1. **Conversion Funnel Drop-off**
   - Visit → Browse (80%)
   - Browse → Cart (40%)
   - Cart → Purchase (60%)

2. **Segment Behaviors**
   - Power shoppers: High pages + High conversion
   - Window shoppers: High pages + Low conversion
   - Quick deciders: Low pages + High conversion
   - Bounces: 1-2 pages + No conversion

3. **Discount Effects**
   - Sweet spot around 15-20% discount
   - Diminishing returns above 25%
   - Some customers convert without discounts

4. **Demographic Patterns**
   - Age and price sensitivity correlation
   - Gender differences in product categories
   - Location affects payment methods

5. **Temporal Patterns**
   - Weekday vs weekend differences
   - Lunch hour and evening peaks
   - Day-of-week effects on conversion

---

## 💾 Loading the Data

### Basic Load
```python
import pandas as pd

df = pd.read_csv('data/consumer_behavior_dataset.csv')
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
```

### Optimized Load (for large datasets)
```python
# Specify data types to reduce memory
dtypes = {
    'purchase_decision': 'int8',
    'age': 'int16',
    'pages_visited': 'int16',
    'cart_added': 'int8',
    'rating': 'int8',
    # ... etc
}

df = pd.read_csv('data/consumer_behavior_dataset.csv', dtype=dtypes)
```

### Using Polars (faster alternative)
```python
import polars as pl

df = pl.read_csv('data/consumer_behavior_dataset.csv')
# Polars is 5-10x faster for large operations
```

---

## 🎓 Domain Knowledge Notes

### E-Commerce Benchmarks
- **Good conversion rate:** 2-5% (industry average)
- **Excellent conversion rate:** 5-10%
- **Cart abandonment:** 60-80% is typical
- **Repeat purchase rate:** 20-30% for most sites

### Key Metrics Definitions
- **Conversion Rate:** (Purchases / Total Sessions) × 100
- **Average Order Value (AOV):** Total Revenue / Number of Orders
- **Cart Abandonment Rate:** (1 - Purchases/Cart Adds) × 100
- **Bounce Rate:** Sessions with 1 page view / Total Sessions

---

## 📝 Data Dictionary Summary

**Total Features:** 19 columns  
**Target Variable:** `purchase_decision` (binary)  
**Feature Types:**
- Numeric: 6-8 columns (price, age, session_duration, etc.)
- Categorical: 8-10 columns (gender, location, product_category, etc.)
- Binary: 3-4 columns (cart_added, cart_abandoned, etc.)
- Text: 1 column (review_text)

**Recommended Train/Test Split:** 80/20 or 70/30  
**Cross-Validation:** 5-fold stratified (preserves class distribution)

---

**Next Step:** Read 03_ANALYSIS_PLAN.md for detailed analysis roadmap!