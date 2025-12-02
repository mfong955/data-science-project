# SQL Queries for Practice

## 🗄️ Database Setup

### Create SQLite Database

```python
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# Load data
df = pd.read_csv('data/consumer_behavior_dataset.csv')

# Create SQLite database
engine = create_engine('sqlite:///ecommerce.db')

# Load into database
df.to_sql('customer_sessions', engine, if_exists='replace', index=False)

print("✓ Database created: ecommerce.db")
print(f"✓ Table 'customer_sessions' created with {len(df)} rows")
```

### Verify Database

```python
# Test connection
conn = sqlite3.connect('ecommerce.db')
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

## 📋 Query Categories

### Level 1: Basic Queries
- SELECT, WHERE, ORDER BY
- Aggregations (COUNT, AVG, SUM)
- GROUP BY

### Level 2: Intermediate Queries
- Multiple JOINs (simulated with self-joins)
- Subqueries
- CASE statements
- HAVING clause

### Level 3: Advanced Queries
- Window functions
- CTEs (Common Table Expressions)
- Complex aggregations
- Performance optimization

---

## 🎯 LEVEL 1: BASIC QUERIES

### Query 1: Overall Conversion Metrics
```sql
-- Calculate key product metrics
SELECT 
    COUNT(*) as total_sessions,
    SUM(purchase_decision) as total_purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate_pct,
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value,
    ROUND(AVG(session_duration), 2) as avg_session_duration_min,
    ROUND(AVG(pages_visited), 2) as avg_pages_per_session
FROM customer_sessions;
```

**Expected Output:**
| total_sessions | total_purchases | conversion_rate_pct | avg_order_value | avg_session_duration_min | avg_pages_per_session |
|---------------|-----------------|---------------------|-----------------|--------------------------|----------------------|
| 10000 | 1500 | 15.00 | 285.50 | 7.15 | 5.8 |

---

### Query 2: Conversion by Product Category
```sql
-- Analyze conversion rate by product category
SELECT 
    product_category,
    COUNT(*) as sessions,
    SUM(purchase_decision) as purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue
FROM customer_sessions
GROUP BY product_category
ORDER BY conversion_rate DESC;
```

**Business Insight:** Identify which categories perform best/worst

---

### Query 3: Cart Abandonment Analysis
```sql
-- Calculate cart abandonment rate
SELECT 
    COUNT(*) as total_sessions,
    SUM(cart_added) as carts_created,
    SUM(CASE WHEN cart_added = 1 AND purchase_decision = 1 THEN 1 ELSE 0 END) as carts_converted,
    SUM(CASE WHEN cart_added = 1 AND purchase_decision = 0 THEN 1 ELSE 0 END) as carts_abandoned,
    ROUND(
        100.0 * SUM(CASE WHEN cart_added = 1 AND purchase_decision = 0 THEN 1 ELSE 0 END) / 
        NULLIF(SUM(cart_added), 0), 2
    ) as cart_abandonment_rate_pct
FROM customer_sessions;
```

---

### Query 4: Demographics Breakdown
```sql
-- Conversion by age group and gender
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
SELECT 
    CASE 
        WHEN discount = 0 THEN 'No Discount'
        WHEN discount <= 10 THEN '1-10%'
        WHEN discount <= 20 THEN '11-20%'
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

## 🎯 LEVEL 2: INTERMEDIATE QUERIES

### Query 6: Engagement Funnel
```sql
-- Create conversion funnel
WITH funnel_stages AS (
    SELECT 
        'Stage 1: All Sessions' as stage,
        1 as stage_order,
        COUNT(*) as users,
        COUNT(*) as previous_stage_users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 2: Browsed (2+ pages)' as stage,
        2 as stage_order,
        SUM(CASE WHEN pages_visited >= 2 THEN 1 ELSE 0 END) as users,
        COUNT(*) as previous_stage_users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 3: Added to Cart' as stage,
        3 as stage_order,
        SUM(cart_added) as users,
        SUM(CASE WHEN pages_visited >= 2 THEN 1 ELSE 0 END) as previous_stage_users
    FROM customer_sessions
    
    UNION ALL
    
    SELECT 
        'Stage 4: Purchased' as stage,
        4 as stage_order,
        SUM(purchase_decision) as users,
        SUM(cart_added) as previous_stage_users
    FROM customer_sessions
)
SELECT 
    stage,
    users,
    ROUND(100.0 * users / LAG(users) OVER (ORDER BY stage_order), 2) as conversion_from_previous,
    ROUND(100.0 * users / FIRST_VALUE(users) OVER (ORDER BY stage_order), 2) as conversion_from_start
FROM funnel_stages
ORDER BY stage_order;
```

**Business Insight:** Identify where users drop off most

---

### Query 7: High-Value vs Low-Value Customers
```sql
-- Compare behaviors of converters vs non-converters
SELECT 
    CASE WHEN purchase_decision = 1 THEN 'Purchasers' ELSE 'Non-Purchasers' END as customer_type,
    COUNT(*) as count,
    ROUND(AVG(pages_visited), 2) as avg_pages,
    ROUND(AVG(session_duration), 2) as avg_duration_min,
    ROUND(AVG(price), 2) as avg_price_interest,
    ROUND(AVG(discount), 2) as avg_discount,
    ROUND(AVG(sentiment_score), 2) as avg_sentiment
FROM customer_sessions
GROUP BY customer_type;
```

---

### Query 8: Time-Based Patterns
```sql
-- Analyze conversion by time of day and day of week
SELECT 
    day_of_week,
    time_of_day,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(session_duration), 2) as avg_duration
FROM customer_sessions
GROUP BY day_of_week, time_of_day
ORDER BY 
    CASE day_of_week
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END,
    CASE time_of_day
        WHEN 'Morning' THEN 1
        WHEN 'Afternoon' THEN 2
        WHEN 'Evening' THEN 3
        WHEN 'Night' THEN 4
    END;
```

---

### Query 9: Customer Segments (SQL-based)
```sql
-- Segment customers by behavior
SELECT 
    CASE 
        WHEN pages_visited >= 10 AND purchase_decision = 1 THEN 'Power Shoppers'
        WHEN pages_visited >= 10 AND purchase_decision = 0 THEN 'Window Shoppers'
        WHEN pages_visited < 5 AND purchase_decision = 1 THEN 'Quick Deciders'
        WHEN discount > 15 THEN 'Deal Seekers'
        ELSE 'Other'
    END as segment,
    COUNT(*) as count,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(AVG(session_duration), 2) as avg_duration
FROM customer_sessions
GROUP BY segment
ORDER BY conversion_rate DESC;
```

---

### Query 10: Subquery - Above Average Performers
```sql
-- Find sessions with above-average engagement that didn't convert
SELECT 
    product_category,
    COUNT(*) as missed_opportunities,
    ROUND(AVG(pages_visited), 2) as avg_pages,
    ROUND(AVG(session_duration), 2) as avg_duration,
    ROUND(AVG(price), 2) as avg_price
FROM customer_sessions
WHERE purchase_decision = 0
    AND pages_visited > (SELECT AVG(pages_visited) FROM customer_sessions)
    AND session_duration > (SELECT AVG(session_duration) FROM customer_sessions)
GROUP BY product_category
ORDER BY missed_opportunities DESC;
```

**Business Insight:** High-intent users who didn't convert - retargeting opportunity

---

## 🎯 LEVEL 3: ADVANCED QUERIES

### Query 11: Moving Averages (Simulated Time Series)
```sql
-- Calculate rolling conversion rate
-- Note: Requires row_number for time series simulation
WITH numbered_sessions AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY age, gender) as session_order
    FROM customer_sessions
)
SELECT 
    session_order,
    purchase_decision,
    ROUND(AVG(purchase_decision) OVER (
        ORDER BY session_order 
        ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
    ) * 100, 2) as rolling_100_conversion_rate
FROM numbered_sessions
WHERE session_order % 100 = 0  -- Sample every 100 sessions
ORDER BY session_order;
```

---

### Query 12: Percentile Analysis
```sql
-- Analyze engagement by percentiles
WITH engagement_percentiles AS (
    SELECT 
        *,
        NTILE(4) OVER (ORDER BY pages_visited) as page_quartile,
        NTILE(4) OVER (ORDER BY session_duration) as duration_quartile
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

### Query 13: Cohort Analysis (Simulated)
```sql
-- Simulate cohort retention analysis
-- Using age as proxy for cohort
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

### Query 14: Complex Aggregation with Multiple CTEs
```sql
-- Comprehensive customer analysis
WITH customer_metrics AS (
    SELECT 
        *,
        pages_visited * session_duration as engagement_score,
        CASE WHEN discount > 0 THEN 1 ELSE 0 END as received_discount
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

### Query 15: Statistical Comparison (A/B Test in SQL)
```sql
-- Compare high discount vs low discount groups
WITH discount_groups AS (
    SELECT 
        CASE WHEN discount >= 15 THEN 'High Discount' ELSE 'Low Discount' END as group_name,
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
    ROUND(conversions * avg_order_value, 2) as total_revenue
FROM group_stats
ORDER BY group_name;
```

---

## 📊 Practice Exercises

### Exercise 1: Custom Metrics
Write a query to calculate:
- Sessions per user (if user_id existed)
- Average time between sessions
- Repeat purchase rate

### Exercise 2: Price Elasticity
Calculate conversion rate across price buckets:
- $0-50
- $50-100
- $100-200
- $200-500
- $500+

### Exercise 3: Segment Performance
Create a query that:
1. Defines 4 customer segments
2. Calculates key metrics per segment
3. Ranks segments by revenue potential

### Exercise 4: Optimization Opportunities
Write a query to identify:
- High-engagement non-converters (retargeting opportunity)
- Low-engagement converters (upsell opportunity)
- Cart abandoners with high intent (recovery opportunity)

---

## 🔄 Using SQL in Python

### Method 1: Using pandas
```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///ecommerce.db')

# Execute query and get results
query = """
SELECT product_category, AVG(purchase_decision) as conv_rate
FROM customer_sessions
GROUP BY product_category
"""

df = pd.read_sql(query, engine)
print(df)
```

### Method 2: Using sqlite3
```python
import sqlite3

conn = sqlite3.connect('ecommerce.db')

# Execute query
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM customer_sessions")
result = cursor.fetchone()

print(f"Total rows: {result[0]}")

conn.close()
```

### Method 3: SQL in Jupyter (Magic Commands)
```python
%load_ext sql
%sql sqlite:///ecommerce.db

%%sql
SELECT * FROM customer_sessions LIMIT 5;
```

---

## 📝 SQL Best Practices

### Formatting
- Use uppercase for SQL keywords
- Indent subqueries and CTEs
- One column per line in SELECT
- Comment complex logic

### Performance
- Use appropriate indexes
- Avoid SELECT *
- Use CTEs for readability
- Filter early (WHERE before JOIN)

### Naming
- Use descriptive aliases
- Consistent naming conventions
- Clear column names in results

---

## ✅ SQL Skills Checklist

By the end of this practice, you should be able to:
- [ ] Write basic SELECT queries with filtering
- [ ] Use GROUP BY and aggregation functions
- [ ] Create CASE statements for categorization
- [ ] Write subqueries
- [ ] Use CTEs (WITH clauses)
- [ ] Apply window functions
- [ ] Calculate running totals and moving averages
- [ ] Perform cohort analysis
- [ ] Simulate A/B tests
- [ ] Optimize query performance

---

## 🎯 Resume Bullet for SQL Skills

*"Developed comprehensive SQL query library for e-commerce analytics, including conversion funnel analysis, customer segmentation, and A/B test comparisons using CTEs, window functions, and complex aggregations"*

---

**Next Step:** Read 05_CODE_TEMPLATES.md for Python implementation!