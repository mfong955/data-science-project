# SQL Queries for Practice

## 🗄️ Database Setup

**Database Location:** `project/data/processed/ecommerce.db`  
**SQL Files Location:** `project/sql/queries/`

### Create SQLite Database

```python
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from pathlib import Path

# Load data from our project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'consumer_behavior_dataset.csv'
DB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'ecommerce.db'

df = pd.read_csv(DATA_PATH)

# Create SQLite database in processed folder
engine = create_engine(f'sqlite:///{DB_PATH}')

# Load into database
df.to_sql('customer_sessions', engine, if_exists='replace', index=False)

print(f"✓ Database created: {DB_PATH}")
print(f"✓ Table 'customer_sessions' created with {len(df)} rows")
```

### Verify Database

```python
# Test connection
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# Preview data
df_test = pd.read_sql("SELECT * FROM customer_sessions LIMIT 5;", conn)
print(df_test)

conn.close()
```

---

## 📋 Actual Column Names

Based on the actual dataset schema:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | text | User identifier |
| `product_id` | text | Product identifier |
| `category` | text | Product category |
| `price` | real | Product price |
| `discount_applied` | real | Discount percentage |
| `payment_method` | text | Payment method |
| `purchase_date` | text | Date of session |
| `pages_visited` | integer | Pages viewed |
| `time_spent` | integer | Session duration |
| `add_to_cart` | integer | Cart addition (0/1) |
| `abandoned_cart` | integer | Cart abandoned (0/1) |
| `rating` | integer | Product rating |
| `review_text` | text | Review content |
| `sentiment_score` | real | Sentiment score |
| `age` | integer | Customer age |
| `gender` | text | Customer gender |
| `income_level` | text | Income category |
| `location` | text | Customer location |
| `purchase_decision` | integer | **TARGET** (0/1) |

---

## 🎯 LEVEL 1: BASIC QUERIES

### Query 1: Overall Conversion Metrics
```sql
-- Calculate key product metrics
-- Save to: project/sql/queries/01_overall_metrics.sql
SELECT 
    COUNT(*) as total_sessions,
    SUM(purchase_decision) as total_purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value,
    ROUND(AVG(time_spent), 2) as avg_session_duration,
    ROUND(AVG(pages_visited), 2) as avg_pages_per_session
FROM customer_sessions;
```

---

### Query 2: Conversion by Product Category
```sql
-- Analyze conversion rate by product category
-- Save to: project/sql/queries/02_conversion_by_category.sql
SELECT 
    category,
    COUNT(*) as sessions,
    SUM(purchase_decision) as purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue
FROM customer_sessions
GROUP BY category
ORDER BY conversion_rate DESC;
```

---

### Query 3: Cart Abandonment Analysis
```sql
-- Calculate cart abandonment rate
-- Save to: project/sql/queries/03_cart_abandonment.sql
SELECT 
    COUNT(*) as total_sessions,
    SUM(add_to_cart) as carts_created,
    SUM(CASE WHEN add_to_cart = 1 AND purchase_decision = 1 THEN 1 ELSE 0 END) as carts_converted,
    SUM(abandoned_cart) as carts_abandoned,
    ROUND(
        100.0 * SUM(abandoned_cart) / NULLIF(SUM(add_to_cart), 0), 2
    ) as cart_abandonment_rate_pct
FROM customer_sessions;
```

---

### Query 4: Demographics Breakdown
```sql
-- Conversion by age group and gender
-- Save to: project/sql/queries/04_demographics.sql
SELECT 
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    gender,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price
FROM customer_sessions
GROUP BY age_group, gender
ORDER BY age_group, gender;
```

---

### Query 5: Discount Impact
```sql
-- Analyze how discounts affect conversion
-- Save to: project/sql/queries/05_discount_impact.sql
SELECT 
    CASE 
        WHEN discount_applied = 0 THEN 'No Discount'
        WHEN discount_applied <= 10 THEN '1-10%'
        WHEN discount_applied <= 20 THEN '11-20%'
        ELSE '20%+'
    END as discount_tier,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value
FROM customer_sessions
GROUP BY discount_tier
ORDER BY 
    CASE discount_tier
        WHEN 'No Discount' THEN 1
        WHEN '1-10%' THEN 2
        WHEN '11-20%' THEN 3
        ELSE 4
    END;
```

---

### Query 6: Income Level Analysis
```sql
-- Conversion by income level
-- Save to: project/sql/queries/06_income_analysis.sql
SELECT 
    income_level,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price_viewed,
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value,
    ROUND(AVG(discount_applied), 2) as avg_discount_used
FROM customer_sessions
GROUP BY income_level
ORDER BY conversion_rate DESC;
```

---

### Query 7: Payment Method Analysis
```sql
-- Conversion by payment method
-- Save to: project/sql/queries/07_payment_method.sql
SELECT 
    payment_method,
    COUNT(*) as sessions,
    SUM(purchase_decision) as purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value
FROM customer_sessions
GROUP BY payment_method
ORDER BY purchases DESC;
```

---

## 🎯 LEVEL 2: INTERMEDIATE QUERIES

### Query 8: Engagement Funnel
```sql
-- Create conversion funnel
-- Save to: project/sql/queries/08_engagement_funnel.sql
WITH funnel_stages AS (
    SELECT 
        'Stage 1: All Sessions' as stage,
        1 as stage_order,
        COUNT(*) as users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 2: Browsed (2+ pages)' as stage,
        2 as stage_order,
        SUM(CASE WHEN pages_visited >= 2 THEN 1 ELSE 0 END) as users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 3: Added to Cart' as stage,
        3 as stage_order,
        SUM(add_to_cart) as users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 4: Purchased' as stage,
        4 as stage_order,
        SUM(purchase_decision) as users
    FROM customer_sessions
)
SELECT 
    stage,
    users,
    ROUND(100.0 * users / FIRST_VALUE(users) OVER (ORDER BY stage_order), 2) as pct_of_total
FROM funnel_stages
ORDER BY stage_order;
```

---

### Query 9: High-Value vs Low-Value Customers
```sql
-- Compare behaviors of converters vs non-converters
-- Save to: project/sql/queries/09_converter_comparison.sql
SELECT 
    CASE WHEN purchase_decision = 1 THEN 'Purchasers' ELSE 'Non-Purchasers' END as customer_type,
    COUNT(*) as count,
    ROUND(AVG(pages_visited), 2) as avg_pages,
    ROUND(AVG(time_spent), 2) as avg_duration,
    ROUND(AVG(price), 2) as avg_price_interest,
    ROUND(AVG(discount_applied), 2) as avg_discount,
    ROUND(AVG(sentiment_score), 2) as avg_sentiment,
    ROUND(AVG(rating), 2) as avg_rating
FROM customer_sessions
GROUP BY customer_type;
```

---

### Query 10: Location Analysis
```sql
-- Conversion by location
-- Save to: project/sql/queries/10_location_analysis.sql
SELECT 
    location,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue
FROM customer_sessions
GROUP BY location
ORDER BY total_revenue DESC
LIMIT 10;
```

---

### Query 11: Sentiment Impact on Conversion
```sql
-- Analyze how sentiment affects purchase
-- Save to: project/sql/queries/11_sentiment_impact.sql
SELECT 
    CASE 
        WHEN sentiment_score < 0 THEN 'Negative'
        WHEN sentiment_score = 0 THEN 'Neutral'
        ELSE 'Positive'
    END as sentiment_category,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(rating), 2) as avg_rating,
    ROUND(AVG(price), 2) as avg_price
FROM customer_sessions
GROUP BY sentiment_category
ORDER BY conversion_rate DESC;
```

---

### Query 12: Customer Segments (SQL-based)
```sql
-- Segment customers by behavior
-- Save to: project/sql/queries/12_customer_segments.sql
SELECT 
    CASE 
        WHEN pages_visited >= 10 AND purchase_decision = 1 THEN 'Power Shoppers'
        WHEN pages_visited >= 10 AND purchase_decision = 0 THEN 'Window Shoppers'
        WHEN pages_visited < 5 AND purchase_decision = 1 THEN 'Quick Deciders'
        WHEN discount_applied > 15 THEN 'Deal Seekers'
        ELSE 'Other'
    END as segment,
    COUNT(*) as count,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(AVG(time_spent), 2) as avg_duration
FROM customer_sessions
GROUP BY segment
ORDER BY conversion_rate DESC;
```

---

## 🎯 LEVEL 3: ADVANCED QUERIES

### Query 13: Percentile Analysis
```sql
-- Analyze engagement by percentiles
-- Save to: project/sql/queries/13_percentile_analysis.sql
WITH engagement_percentiles AS (
    SELECT 
        *,
        NTILE(4) OVER (ORDER BY pages_visited) as page_quartile,
        NTILE(4) OVER (ORDER BY time_spent) as duration_quartile
    FROM customer_sessions
)
SELECT 
    page_quartile as engagement_quartile,
    ROUND(MIN(pages_visited), 2) as min_pages,
    ROUND(MAX(pages_visited), 2) as max_pages,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate
FROM engagement_percentiles
GROUP BY page_quartile
ORDER BY page_quartile;
```

---

### Query 14: Cohort Analysis by Age
```sql
-- Cohort analysis by age group
-- Save to: project/sql/queries/14_cohort_analysis.sql
WITH cohorts AS (
    SELECT 
        CASE 
            WHEN age < 30 THEN 'Cohort A (18-29)'
            WHEN age < 40 THEN 'Cohort B (30-39)'
            WHEN age < 50 THEN 'Cohort C (40-49)'
            ELSE 'Cohort D (50+)'
        END as cohort,
        *
    FROM customer_sessions
)
SELECT 
    cohort,
    COUNT(*) as cohort_size,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END) / 
          NULLIF(COUNT(*), 0), 2) as revenue_per_user,
    ROUND(AVG(pages_visited), 2) as avg_engagement
FROM cohorts
GROUP BY cohort
ORDER BY cohort;
```

---

### Query 15: Complex Aggregation with Multiple CTEs
```sql
-- Comprehensive customer analysis
-- Save to: project/sql/queries/15_comprehensive_analysis.sql
WITH customer_metrics AS (
    SELECT 
        *,
        pages_visited * time_spent as engagement_score,
        CASE WHEN discount_applied > 0 THEN 1 ELSE 0 END as received_discount
    FROM customer_sessions
),
segment_stats AS (
    SELECT 
        CASE 
            WHEN engagement_score > 100 AND purchase_decision = 1 THEN 'High Value'
            WHEN engagement_score > 100 THEN 'High Potential'
            WHEN purchase_decision = 1 THEN 'Converter'
            ELSE 'Low Engagement'
        END as segment,
        COUNT(*) as count,
        AVG(purchase_decision) as conv_rate,
        AVG(price) as avg_price,
        AVG(engagement_score) as avg_engagement,
        AVG(received_discount) as discount_rate
    FROM customer_metrics
    GROUP BY segment
)
SELECT 
    segment,
    count,
    ROUND(conv_rate * 100, 2) as conversion_rate,
    ROUND(avg_price, 2) as avg_price,
    ROUND(avg_engagement, 2) as avg_engagement,
    ROUND(discount_rate * 100, 2) as pct_receiving_discount,
    ROUND(count * 100.0 / SUM(count) OVER (), 2) as pct_of_total
FROM segment_stats
ORDER BY conv_rate DESC;
```

---

### Query 16: A/B Test Comparison (Discount Groups)
```sql
-- Compare high discount vs low discount groups
-- Save to: project/sql/queries/16_ab_test_comparison.sql
WITH discount_groups AS (
    SELECT 
        CASE WHEN discount_applied >= 15 THEN 'High Discount' ELSE 'Low Discount' END as group_name,
        purchase_decision,
        price
    FROM customer_sessions
),
group_stats AS (
    SELECT 
        group_name,
        COUNT(*) as n,
        SUM(purchase_decision) as conversions,
        AVG(purchase_decision) as conv_rate,
        AVG(CASE WHEN purchase_decision = 1 THEN price END) as avg_order_value
    FROM discount_groups
    GROUP BY group_name
)
SELECT 
    group_name,
    n,
    conversions,
    ROUND(conv_rate * 100, 2) as conversion_rate_pct,
    ROUND(avg_order_value, 2) as avg_order_value,
    ROUND(conversions * COALESCE(avg_order_value, 0), 2) as total_revenue
FROM group_stats
ORDER BY group_name;
```

---

## 📊 Practice Exercises

### Exercise 1: Category Performance
Write a query to find:
- Top 3 categories by conversion rate
- Bottom 3 categories by conversion rate
- Revenue contribution by category

### Exercise 2: Price Elasticity
Calculate conversion rate across price buckets:
- $0-50
- $50-100
- $100-200
- $200-500
- $500+

### Exercise 3: Engagement Scoring
Create a query that:
1. Calculates engagement score (pages × time)
2. Segments users into Low/Medium/High engagement
3. Compares conversion rates across segments

### Exercise 4: Cart Recovery Opportunities
Write a query to identify:
- Users who added to cart but didn't purchase
- Their average price point
- Their sentiment scores
- Potential recovery value

---

## 🔄 Using SQL in Python

### Method 1: Using pandas
```python
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'ecommerce.db'

engine = create_engine(f'sqlite:///{DB_PATH}')

query = """
SELECT category, AVG(purchase_decision) as conv_rate
FROM customer_sessions
GROUP BY category
"""

df = pd.read_sql(query, engine)
print(df)
```

### Method 2: Using sqlite3
```python
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'ecommerce.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM customer_sessions")
result = cursor.fetchone()
print(f"Total rows: {result[0]}")
conn.close()
```

---

## 📁 Where to Save SQL Files

Save your SQL queries to: `project/sql/queries/`

Suggested naming:
- `01_overall_metrics.sql`
- `02_conversion_by_category.sql`
- `03_cart_abandonment.sql`
- etc.

---

## ✅ SQL Skills Checklist

By the end of this practice, you should be able to:
- [ ] Write basic SELECT queries with filtering
- [ ] Use GROUP BY and aggregation functions
- [ ] Create CASE statements for categorization
- [ ] Write subqueries
- [ ] Use CTEs (WITH clauses)
- [ ] Apply window functions
- [ ] Calculate percentiles with NTILE
- [ ] Perform cohort analysis
- [ ] Simulate A/B tests
- [ ] Optimize query performance

---

**Next Step:** Read 05_CODE_TEMPLATES.md for Python implementation!