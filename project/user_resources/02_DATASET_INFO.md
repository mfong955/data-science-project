# Dataset Information

## 📊 AI-Driven Consumer Behavior Dataset

**Source:** Kaggle - AI-Driven Consumer Behavior Dataset  
**Location:** `project/data/raw/consumer_behavior_dataset.csv`  
**Size:** 5,000 rows × 19 columns  
**Target Variable:** `purchase_decision` (binary: 0/1)

---

## 📋 Column Schema (Actual)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | object | Unique identifier for each user |
| `product_id` | object | Unique identifier for each product |
| `category` | object | Product category |
| `price` | float64 | Product price |
| `discount_applied` | float64 | Discount percentage applied |
| `payment_method` | object | Payment method used |
| `purchase_date` | object | Date of purchase/session |
| `pages_visited` | int64 | Number of pages visited in session |
| `time_spent` | int64 | Time spent on site (likely minutes) |
| `add_to_cart` | int64 | Whether item was added to cart (0/1) |
| `abandoned_cart` | int64 | Whether cart was abandoned (0/1) |
| `rating` | int64 | Product rating given |
| `review_text` | object | Text of review (if any) |
| `sentiment_score` | float64 | Sentiment score of review |
| `age` | int64 | Customer age |
| `gender` | object | Customer gender |
| `income_level` | object | Customer income level category |
| `location` | object | Customer location |
| `purchase_decision` | int64 | **TARGET** - Whether purchase was made (0/1) |

---

## 🔍 Column Categories

### Identifiers
- `user_id` - User identifier
- `product_id` - Product identifier

### Product Features
- `category` - Product category
- `price` - Product price
- `discount_applied` - Discount percentage

### Transaction Features
- `payment_method` - Payment method
- `purchase_date` - Date of transaction/session

### Behavioral Features (Session)
- `pages_visited` - Pages viewed in session
- `time_spent` - Session duration
- `add_to_cart` - Cart addition flag
- `abandoned_cart` - Cart abandonment flag

### Review Features
- `rating` - Product rating
- `review_text` - Review text content
- `sentiment_score` - Computed sentiment

### Demographic Features
- `age` - Customer age
- `gender` - Customer gender
- `income_level` - Income category
- `location` - Geographic location

### Target Variable
- `purchase_decision` - Binary outcome (0 = no purchase, 1 = purchase)

---

## 📈 Quick Data Loading

```python
import pandas as pd
from pathlib import Path

# From notebooks/exploratory/ folder
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'consumer_behavior_dataset.csv'

# Or from project root
# DATA_PATH = 'project/data/raw/consumer_behavior_dataset.csv'

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.info())
```

---

## 🎯 Key Metrics to Calculate

### Conversion Metrics
```python
# Conversion rate
conversion_rate = df['purchase_decision'].mean()
print(f"Conversion Rate: {conversion_rate:.2%}")

# Cart abandonment rate
cart_users = df[df['add_to_cart'] == 1]
abandonment_rate = cart_users['abandoned_cart'].mean()
print(f"Cart Abandonment Rate: {abandonment_rate:.2%}")
```

### Revenue Metrics
```python
# Average order value (for purchasers)
purchasers = df[df['purchase_decision'] == 1]
avg_order_value = purchasers['price'].mean()
print(f"Average Order Value: ${avg_order_value:.2f}")

# Average discount applied
avg_discount = df['discount_applied'].mean()
print(f"Average Discount: {avg_discount:.1f}%")
```

### Engagement Metrics
```python
# Average pages per session
avg_pages = df['pages_visited'].mean()
print(f"Avg Pages/Session: {avg_pages:.1f}")

# Average time spent
avg_time = df['time_spent'].mean()
print(f"Avg Time Spent: {avg_time:.1f} min")
```

---

## 🔄 Column Mapping (Old → New)

If you see references to old column names in documentation, here's the mapping:

| Old Name (Documented) | Actual Name |
|----------------------|-------------|
| `session_id` | `user_id` |
| `product_category` | `category` |
| `discount` | `discount_applied` |
| `session_duration` | `time_spent` |
| `cart_added` | `add_to_cart` |
| `wishlist_added` | N/A (not in dataset) |
| `reviews_viewed` | N/A (not in dataset) |
| `device_type` | N/A (not in dataset) |
| `browser` | N/A (not in dataset) |
| `time_of_day` | N/A (extract from `purchase_date`) |
| `day_of_week` | N/A (extract from `purchase_date`) |

---

## 🛠️ Feature Engineering Opportunities

### From `purchase_date`
```python
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['day_of_week'] = df['purchase_date'].dt.day_name()
df['month'] = df['purchase_date'].dt.month
df['is_weekend'] = df['purchase_date'].dt.dayofweek >= 5
```

### Engagement Score
```python
df['engagement_score'] = df['pages_visited'] * df['time_spent']
```

### Price Features
```python
df['price_after_discount'] = df['price'] * (1 - df['discount_applied'] / 100)
df['discount_amount'] = df['price'] * df['discount_applied'] / 100
```

### Behavioral Flags
```python
df['high_engagement'] = (df['pages_visited'] > df['pages_visited'].median()).astype(int)
df['cart_converted'] = ((df['add_to_cart'] == 1) & (df['purchase_decision'] == 1)).astype(int)
```

---

## 📊 Expected Distributions

### Target Variable
- `purchase_decision`: Likely imbalanced (more 0s than 1s)
- Check with: `df['purchase_decision'].value_counts(normalize=True)`

### Categorical Variables
- `category`: Multiple product categories
- `gender`: Male/Female/Other
- `income_level`: Low/Medium/High or similar
- `payment_method`: Credit Card, PayPal, etc.
- `location`: Various geographic locations

### Numeric Variables
- `price`: Continuous, likely right-skewed
- `age`: Integer, likely 18-70+ range
- `pages_visited`: Integer, likely 1-20+ range
- `time_spent`: Integer, session duration
- `sentiment_score`: Float, likely -1 to 1 or 0 to 1

---

## ✅ Data Quality Checks

```python
# Missing values
print("Missing Values:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Unique values for categoricals
for col in ['category', 'gender', 'income_level', 'payment_method', 'location']:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())
```

---

## 📁 File Locations

- **Raw Data:** `project/data/raw/consumer_behavior_dataset.csv`
- **Processed Data:** `project/data/processed/` (after cleaning)
- **SQLite Database:** `project/data/processed/ecommerce.db` (after creation)

---

**Next Step:** Read 03_ANALYSIS_PLAN.md for the analysis roadmap!