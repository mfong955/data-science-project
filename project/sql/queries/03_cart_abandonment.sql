-- ============================================================================
-- Query 3: Cart Abandonment Analysis
-- ============================================================================
-- Purpose: Calculate cart abandonment rate and related metrics
-- Level: Basic (Level 1)
-- 
-- SQL Concepts Used:
--   NULLIF(value, 0) - Returns NULL if value equals 0 (prevents division by zero)
--   Nested CASE statements - Multiple conditions in one expression
--   Division with NULL handling - Safe division patterns
-- ============================================================================

SELECT 
    -- Total sessions in the dataset
    COUNT(*) as total_sessions,
    
    -- How many users added items to cart
    -- SUM on a 0/1 column counts the 1s
    SUM(add_to_cart) as carts_created,
    
    -- How many cart users actually purchased
    -- This uses a compound condition: added to cart AND purchased
    SUM(CASE WHEN add_to_cart = 1 AND purchase_decision = 1 THEN 1 ELSE 0 END) as carts_converted,
    
    -- How many carts were abandoned
    SUM(abandoned_cart) as carts_abandoned,
    
    -- Cart abandonment rate calculation
    -- NULLIF prevents division by zero by returning NULL if denominator is 0
    -- Formula: abandoned_carts / total_carts * 100
    ROUND(
        100.0 * SUM(abandoned_cart) / NULLIF(SUM(add_to_cart), 0), 
        2
    ) as cart_abandonment_rate_pct,
    
    -- Cart conversion rate (opposite of abandonment)
    ROUND(
        100.0 * SUM(CASE WHEN add_to_cart = 1 AND purchase_decision = 1 THEN 1 ELSE 0 END) 
        / NULLIF(SUM(add_to_cart), 0), 
        2
    ) as cart_conversion_rate_pct

FROM customer_sessions;

-- ============================================================================
-- BONUS: Cart abandonment by category
-- ============================================================================
-- Uncomment to run this additional analysis

/*
SELECT 
    category,
    SUM(add_to_cart) as carts_created,
    SUM(abandoned_cart) as carts_abandoned,
    ROUND(
        100.0 * SUM(abandoned_cart) / NULLIF(SUM(add_to_cart), 0), 
        2
    ) as abandonment_rate_pct
FROM customer_sessions
WHERE add_to_cart = 1  -- Only look at sessions where cart was created
GROUP BY category
ORDER BY abandonment_rate_pct DESC;
*/
