-- ============================================================================
-- Query 8: Engagement Funnel Analysis
-- ============================================================================
-- Purpose: Create a conversion funnel showing user progression through stages
-- Level: Intermediate (Level 2)
-- 
-- SQL Concepts Used:
--   WITH clause (CTE) - Common Table Expression for readable subqueries
--   UNION ALL - Combines multiple SELECT statements into one result
--   Window Functions - FIRST_VALUE() to reference values across rows
--   OVER() clause - Defines the window for window functions
-- ============================================================================

-- CTE (Common Table Expression) - Think of it as a temporary named result set
-- WITH creates a "virtual table" that exists only for this query
-- CTEs make complex queries more readable by breaking them into logical steps
WITH funnel_stages AS (
    -- Stage 1: All sessions (everyone who visited)
    SELECT 
        'Stage 1: All Sessions' as stage,
        1 as stage_order,  -- Used for sorting later
        COUNT(*) as users
    FROM customer_sessions
    
    -- UNION ALL combines results from multiple SELECT statements
    -- UNION ALL keeps all rows; UNION would remove duplicates
    UNION ALL
    
    -- Stage 2: Users who browsed (viewed 2+ pages)
    SELECT 
        'Stage 2: Browsed (2+ pages)' as stage,
        2 as stage_order,
        SUM(CASE WHEN pages_visited >= 2 THEN 1 ELSE 0 END) as users
    FROM customer_sessions
    
    UNION ALL
    
    -- Stage 3: Users who added to cart
    SELECT 
        'Stage 3: Added to Cart' as stage,
        3 as stage_order,
        SUM(add_to_cart) as users
    FROM customer_sessions
    
    UNION ALL
    
    -- Stage 4: Users who purchased
    SELECT 
        'Stage 4: Purchased' as stage,
        4 as stage_order,
        SUM(purchase_decision) as users
    FROM customer_sessions
)

-- Main query uses the CTE
SELECT 
    stage,
    users,
    
    -- FIRST_VALUE() is a window function that returns the first value in the window
    -- OVER (ORDER BY stage_order) defines the window as all rows ordered by stage_order
    -- This gets the "All Sessions" count to calculate percentage of total
    ROUND(
        100.0 * users / FIRST_VALUE(users) OVER (ORDER BY stage_order), 
        2
    ) as pct_of_total,
    
    -- Calculate drop-off from previous stage
    -- LAG() gets the value from the previous row
    ROUND(
        100.0 * users / LAG(users) OVER (ORDER BY stage_order),
        2
    ) as pct_of_previous_stage

FROM funnel_stages
ORDER BY stage_order;

-- ============================================================================
-- EXPECTED OUTPUT:
-- Stage                      | users | pct_of_total | pct_of_previous_stage
-- Stage 1: All Sessions      | 10000 | 100.00       | NULL
-- Stage 2: Browsed (2+ pages)| 8500  | 85.00        | 85.00
-- Stage 3: Added to Cart     | 4000  | 40.00        | 47.06
-- Stage 4: Purchased         | 1500  | 15.00        | 37.50
--
-- BUSINESS INSIGHT:
-- - Where is the biggest drop-off? (Biggest opportunity for improvement)
-- - What percentage of cart-adders actually purchase?
-- ============================================================================
