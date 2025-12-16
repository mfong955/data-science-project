-- ============================================================================
-- Query 1: Overall Conversion Metrics
-- ============================================================================
-- Purpose: Calculate key product metrics for the entire dataset
-- Level: Basic (Level 1)
-- 
-- SQL Concepts Used:
--   COUNT(*) - Counts all rows in the result set
--   SUM() - Adds up all values in a column
--   AVG() - Calculates the arithmetic mean
--   ROUND(value, decimals) - Rounds a number to specified decimal places
--   CASE WHEN - Conditional logic (like if-else in programming)
-- ============================================================================

SELECT 
    -- COUNT(*) counts every row in the table
    COUNT(*) as total_sessions,
    
    -- SUM(purchase_decision) adds up all 1s (since it's binary 0/1)
    -- This gives us the total number of purchases
    SUM(purchase_decision) as total_purchases,
    
    -- AVG(purchase_decision) on a 0/1 column gives the proportion of 1s
    -- Multiply by 100 to get percentage, ROUND to 2 decimal places
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate_pct,
    
    -- CASE WHEN filters the AVG to only include rows where purchase_decision = 1
    -- This calculates Average Order Value (AOV) for actual purchasers only
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value,
    
    -- Simple averages for engagement metrics
    ROUND(AVG(time_spent), 2) as avg_session_duration,
    ROUND(AVG(pages_visited), 2) as avg_pages_per_session

FROM customer_sessions;
