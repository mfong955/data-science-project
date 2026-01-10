# Learning Guide - Project Files Overview

This guide helps you understand what each file does, what you can learn from it, and the recommended order to study them.

---

## 📚 Recommended Learning Path

### Phase 1: Python + Pandas Fundamentals (Start Here)
**Goal:** Learn data manipulation and visualization basics

| Order | File | What You'll Learn |
|-------|------|-------------------|
| 1 | [`explore_raw.py`](../../notebooks/exploratory/explore_raw.py) | Pandas basics, data loading, EDA, matplotlib/seaborn |

**Key concepts in this file:**
- Lines 1-16: Imports and path setup
- Lines 28-92: Basic pandas operations (`.mean()`, `.describe()`, filtering)
- Lines 162-223: Univariate/Bivariate analysis with visualizations
- Lines 224-373: Advanced analysis with detailed comments

**Practice exercises:**
1. Run each section interactively and observe the output
2. Modify the visualizations (change colors, titles, bin sizes)
3. Add new analyses (e.g., conversion by location)

---

### Phase 2: SQL Fundamentals
**Goal:** Learn SQL from basic to advanced

#### Step 1: Database Setup
| File | Purpose |
|------|---------|
| [`setup_database.py`](../../sql/setup_database.py) | Creates SQLite database from CSV |

**What you'll learn:**
- `sqlite3.connect()` - Database connections
- `df.to_sql()` - Writing DataFrames to SQL
- `CREATE INDEX` - Performance optimization
- `pd.read_sql()` - Reading SQL results into pandas

**Run this first:**
```python
exec(open('project/sql/setup_database.py').read())
```

---

#### Step 2: Basic SQL Queries (Level 1)
Study these in order - each builds on the previous:

| Order | File | SQL Concepts |
|-------|------|--------------|
| 1 | [`01_overall_metrics.sql`](../../sql/queries/01_overall_metrics.sql) | `SELECT`, `COUNT`, `SUM`, `AVG`, `ROUND`, `CASE WHEN` |
| 2 | [`02_conversion_by_category.sql`](../../sql/queries/02_conversion_by_category.sql) | `GROUP BY`, `ORDER BY`, aggregate functions |
| 3 | [`03_cart_abandonment.sql`](../../sql/queries/03_cart_abandonment.sql) | `NULLIF` (safe division), compound conditions |
| 4 | [`04_demographics.sql`](../../sql/queries/04_demographics.sql) | Age bucketing with `CASE WHEN`, multiple `GROUP BY` |
| 5 | [`05_discount_impact.sql`](../../sql/queries/05_discount_impact.sql) | Custom `ORDER BY` with `CASE` |

**Practice:**
1. Read each `.sql` file - comments explain every concept
2. Run queries using `run_queries.py` or copy into your interactive session
3. Modify queries to answer your own questions

---

#### Step 3: Intermediate SQL (Level 2)
| Order | File | SQL Concepts |
|-------|------|--------------|
| 6 | [`08_engagement_funnel.sql`](../../sql/queries/08_engagement_funnel.sql) | CTEs (`WITH` clause), `UNION ALL`, Window Functions (`FIRST_VALUE`, `LAG`) |
| 7 | [`12_customer_segments.sql`](../../sql/queries/12_customer_segments.sql) | Complex `CASE` logic, subqueries in `SELECT` |

**Key learning:** CTEs make complex queries readable by breaking them into steps.

---

#### Step 4: Advanced SQL (Level 3)
| Order | File | SQL Concepts |
|-------|------|--------------|
| 8 | [`13_percentile_analysis.sql`](../../sql/queries/13_percentile_analysis.sql) | `NTILE()` for quartiles/deciles, multiple CTEs |
| 9 | [`15_comprehensive_analysis.sql`](../../sql/queries/15_comprehensive_analysis.sql) | Chained CTEs, `SUM() OVER()`, derived columns |
| 10 | [`16_ab_test_comparison.sql`](../../sql/queries/16_ab_test_comparison.sql) | A/B test simulation, confidence intervals, `COALESCE` |

---

### Phase 3: Running Everything Together
| File | Purpose |
|------|---------|
| [`run_queries.py`](../../sql/run_queries.py) | Executes all SQL queries and displays results |

**What you'll learn:**
- Reading SQL files from Python
- Helper functions for database operations
- Combining SQL + Python workflows

---

## 🏭 Production vs Learning Files

### Learning Files (Study These)
These files have extensive comments explaining concepts:

| File | Type | Comments Level |
|------|------|----------------|
| `explore_raw.py` | Python/Pandas | ⭐⭐⭐ Heavy comments from line 224 |
| `setup_database.py` | Python/SQL | ⭐⭐⭐ Every method explained |
| `01-05_*.sql` | SQL Basic | ⭐⭐⭐ Concept explanations |
| `08, 12_*.sql` | SQL Intermediate | ⭐⭐⭐ CTE and window function explanations |
| `13, 15, 16_*.sql` | SQL Advanced | ⭐⭐⭐ Advanced patterns explained |

### Production-Style Files (Reference)
These would be cleaner in production (fewer comments, more modular):

| File | Production Notes |
|------|------------------|
| `run_queries.py` | Good example of helper functions and modular code |
| SQL query files | In production, these would be in a query library or ORM |

---

## 📖 Concept Reference

### Python/Pandas Concepts (in `explore_raw.py`)

| Line | Concept | Description |
|------|---------|-------------|
| 17-21 | `Path.mkdir()` | Create directories programmatically |
| 207-212 | `plt.savefig()` | Save figures to files |
| 233 | `pd.cut()` | Bin continuous data into categories |
| 238 | `.agg()` | Apply multiple aggregations at once |
| 283 | `.select_dtypes()` | Filter columns by data type |
| 289 | `.corr()` | Correlation matrix |
| 304 | `sns.heatmap()` | Seaborn heatmap visualization |

### SQL Concepts (in query files)

| Query | Concept | Description |
|-------|---------|-------------|
| 01 | `CASE WHEN` | Conditional logic in SQL |
| 02 | `GROUP BY` | Aggregate by categories |
| 03 | `NULLIF` | Prevent division by zero |
| 08 | `WITH` (CTE) | Common Table Expressions |
| 08 | `UNION ALL` | Combine result sets |
| 08 | `FIRST_VALUE()` | Window function - first value |
| 08 | `LAG()` | Window function - previous row |
| 13 | `NTILE()` | Divide into equal groups |
| 15 | `SUM() OVER()` | Running totals |
| 16 | Confidence Intervals | Statistical analysis in SQL |

---

## 🎯 Suggested Practice Exercises

### Beginner
1. In `explore_raw.py`, add a visualization for conversion by location
2. Write a SQL query to find the top 5 products by revenue
3. Modify `04_demographics.sql` to use different age brackets

### Intermediate
1. Create a new SQL query that combines category and income level analysis
2. Add a new funnel stage to `08_engagement_funnel.sql`
3. Create a Python function that runs any SQL file and returns a DataFrame

### Advanced
1. Write a SQL query using window functions to calculate week-over-week conversion change
2. Create a cohort analysis by signup week
3. Build a simple A/B test significance calculator in Python using the SQL output

---

## 📁 File Organization Summary

```
project/
├── notebooks/exploratory/
│   └── explore_raw.py          # 📚 START HERE - Python/Pandas learning
│
├── sql/
│   ├── setup_database.py       # 📚 Database setup with comments
│   ├── run_queries.py          # 🏭 Production-style query runner
│   └── queries/
│       ├── 01-05_*.sql         # 📚 Basic SQL (Level 1)
│       ├── 08, 12_*.sql        # 📚 Intermediate SQL (Level 2)
│       └── 13, 15, 16_*.sql    # 📚 Advanced SQL (Level 3)
│
├── data/
│   ├── raw/                    # Source data (CSV)
│   └── processed/              # Database (ecommerce.db)
│
└── visualizations/figures/     # Saved plots from explore_raw.py
```

---

## ⏱️ Estimated Learning Time

| Phase | Time | Focus |
|-------|------|-------|
| Phase 1: Python/Pandas | 2-3 hours | `explore_raw.py` |
| Phase 2: SQL Basics | 2-3 hours | Queries 01-05 |
| Phase 2: SQL Intermediate | 2-3 hours | Queries 08, 12 |
| Phase 2: SQL Advanced | 3-4 hours | Queries 13, 15, 16 |
| Phase 3: Integration | 1-2 hours | `run_queries.py` |

**Total: ~10-15 hours** for thorough understanding

---

## 🚀 Next Steps After This Project

1. **Customer Segmentation** (Analysis 2 in plan) - K-means clustering
2. **Predictive Modeling** (Analysis 3) - Scikit-learn, XGBoost
3. **A/B Testing** (Analysis 4) - Statistical hypothesis testing
4. **Dashboard** (Analysis 5) - Streamlit or similar

See [`03_ANALYSIS_PLAN.md`](../../user_resources/03_ANALYSIS_PLAN.md) for the full project roadmap!
