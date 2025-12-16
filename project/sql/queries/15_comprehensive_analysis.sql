-- ============================================================================
-- Query 15: Comprehensive Customer Analysis with Multiple CTEs
-- ============================================================================
-- Purpose: Demonstrate advanced SQL with multiple CTEs and complex aggregations
-- Level: Advanced (Level 3)
-- 
-- SQL Concepts Used:
--   Multiple CTEs - Chaining several temporary result sets
--   Derived columns in CTEs - Creating calculated fields for later use
--   Complex segmentation - Multi-factor customer classification
--   Window functions with SUM() OVER() - Running totals and percentages
--   COALESCE() - Handle NULL values with default
-- ============================================================================

-- CTE 1: Add derived metrics to each row
WITH customer_metrics AS (
    SELECT 
        *,
        -- Engagement score: pages × time (higher = more engaged)
        pages_visited * time_spent as engagement_score,
        
        -- Binary flag for discount usage
        CASE WHEN discount_applied > 0 THEN 1 ELSE 0 END as received_discount,
        
        -- Price tier classification
        CASE 
            WHEN price < 100 THEN 'Low'
            WHEN price < 300 THEN 'Medium'
            ELSE 'High'
        END as price_tier,
        
        -- Engagement level classification
        CASE 
            WHEN pages_visited * time_spent > 200 THEN 'High'
            WHEN pages_visited * time_spent > 50 THEN 'Medium'
            ELSE 'Low'
        END as engagement_level
    FROM customer_sessions
),

-- CTE 2: Create customer segments based on multiple factors
segment_assignment AS (
    SELECT 
        *,
        CASE 
            -- High Value: High engagement AND purchased
            WHEN engagement_score > 100 AND purchase_decision = 1 THEN 'High Value'
            
            -- High Potential: High engagement but didn't purchase (opportunity!)
            WHEN engagement_score > 100 THEN 'High Potential'
            
            -- Converter: Lower engagement but still purchased
            WHEN purchase_decision = 1 THEN 'Converter'
            
            -- Low Engagement: Everyone else
            ELSE 'Low Engagement'
        END as segment
    FROM customer_metrics
),

-- CTE 3: Calculate segment-level statistics
segment_stats AS (
    SELECT 
        segment,
        COUNT(*) as count,
        AVG(purchase_decision) as conv_rate,
        AVG(price) as avg_price,
        AVG(engagement_score) as avg_engagement,
        AVG(received_discount) as discount_rate,
        SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END) as total_revenue
    FROM segment_assignment
    GROUP BY segment
)

-- Final output with additional calculations
SELECT 
    segment,
    count,
    
    -- Conversion rate as percentage
    ROUND(conv_rate * 100, 2) as conversion_rate,
    
    -- Average metrics
    ROUND(avg_price, 2) as avg_price,
    ROUND(avg_engagement, 2) as avg_engagement,
    ROUND(discount_rate * 100, 2) as pct_receiving_discount,
    
    -- Revenue metrics
    ROUND(total_revenue, 2) as total_revenue,
    
    -- Percentage of total customers in this segment
    -- SUM() OVER () calculates sum across ALL rows (no partition)
    ROUND(count * 100.0 / SUM(count) OVER (), 2) as pct_of_customers,
    
    -- Percentage of total revenue from this segment
    ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER (), 2) as pct_of_revenue,
    
    -- Revenue per customer in segment
    ROUND(total_revenue / NULLIF(count, 0), 2) as revenue_per_customer

FROM segment_stats
ORDER BY conv_rate DESC;

-- ============================================================================
-- ADVANCED ANALYSIS: Cross-tabulation of segments
-- ============================================================================

/*
-- This shows how segments break down by other dimensions
WITH customer_metrics AS (
    SELECT 
        *,
        pages_visited * time_spent as engagement_score,
        CASE 
            WHEN pages_visited * time_spent > 100 AND purchase_decision = 1 THEN 'High Value'
            WHEN pages_visited * time_spent > 100 THEN 'High Potential'
            WHEN purchase_decision = 1 THEN 'Converter'
            ELSE 'Low Engagement'
        END as segment
    FROM customer_sessions
)
SELECT 
    segment,
    income_level,
    COUNT(*) as count,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate
FROM customer_metrics
GROUP BY segment, income_level
ORDER BY segment, income_level;
*/

-- ============================================================================
-- BUSINESS INSIGHTS:
-- 
-- This query helps answer:
-- 1. What % of customers are in each segment?
-- 2. Which segment generates the most revenue?
-- 3. Is there a mismatch between customer count and revenue contribution?
--    (e.g., 20% of customers generating 80% of revenue)
-- 4. Which segment has the highest revenue per customer?
-- 5. Are discounts more common in certain segments?
-- ============================================================================
