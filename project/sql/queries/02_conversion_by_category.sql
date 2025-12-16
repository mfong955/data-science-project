-- ============================================================================
-- Query 2: Conversion by Product Category
-- ============================================================================
-- Purpose: Analyze conversion rate and revenue by product category
-- Level: Basic (Level 1)
-- 
-- SQL Concepts Used:
--   GROUP BY - Groups rows that have the same values in specified columns
--   ORDER BY - Sorts the result set by one or more columns
--   DESC - Sorts in descending order (highest first)
--   Aggregate functions with GROUP BY - Calculates values per group
-- ============================================================================

SELECT 
    -- The column we're grouping by
    category,
    
    -- COUNT(*) now counts rows PER category (because of GROUP BY)
    COUNT(*) as sessions,
    
    -- SUM counts purchases per category
    SUM(purchase_decision) as purchases,
    
    -- AVG calculates conversion rate per category
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    
    -- Average price viewed in each category
    ROUND(AVG(price), 2) as avg_price,
    
    -- Total revenue per category
    -- CASE WHEN ensures we only sum prices where a purchase was made
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue

FROM customer_sessions

-- GROUP BY tells SQL to aggregate data by unique values in 'category'
-- Without GROUP BY, we'd get one row for the entire table
-- With GROUP BY category, we get one row per unique category
GROUP BY category

-- ORDER BY sorts results; DESC means highest conversion_rate first
ORDER BY conversion_rate DESC;
