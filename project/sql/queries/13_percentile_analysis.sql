-- ============================================================================
-- Query 13: Percentile Analysis with Window Functions
-- ============================================================================
-- Purpose: Analyze engagement by percentiles using advanced window functions
-- Level: Advanced (Level 3)
-- 
-- SQL Concepts Used:
--   NTILE() - Divides rows into N equal groups (quartiles, deciles, etc.)
--   Window Functions with OVER() - Calculations across row sets
--   Multiple CTEs - Chaining temporary result sets
--   Percentile-based analysis - Statistical segmentation
-- ============================================================================

-- First CTE: Add percentile rankings to each row
WITH engagement_percentiles AS (
    SELECT 
        *,
        -- NTILE(4) divides all rows into 4 equal groups (quartiles)
        -- OVER (ORDER BY pages_visited) determines the ordering for division
        -- Quartile 1 = lowest 25%, Quartile 4 = highest 25%
        NTILE(4) OVER (ORDER BY pages_visited) as page_quartile,
        
        -- Same for time spent
        NTILE(4) OVER (ORDER BY time_spent) as duration_quartile,
        
        -- NTILE(10) creates deciles (10 groups of 10% each)
        NTILE(10) OVER (ORDER BY pages_visited * time_spent) as engagement_decile
    FROM customer_sessions
),

-- Second CTE: Calculate stats per quartile
quartile_stats AS (
    SELECT 
        page_quartile as engagement_quartile,
        
        -- Range of pages in this quartile
        MIN(pages_visited) as min_pages,
        MAX(pages_visited) as max_pages,
        
        -- Count and conversion
        COUNT(*) as sessions,
        ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
        
        -- Revenue metrics
        ROUND(AVG(price), 2) as avg_price,
        ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue
    FROM engagement_percentiles
    GROUP BY page_quartile
)

-- Final output
SELECT 
    engagement_quartile,
    min_pages,
    max_pages,
    sessions,
    conversion_rate,
    avg_price,
    total_revenue,
    
    -- Calculate percentage of total revenue
    ROUND(
        total_revenue * 100.0 / SUM(total_revenue) OVER (), 
        2
    ) as pct_of_revenue

FROM quartile_stats
ORDER BY engagement_quartile;

-- ============================================================================
-- BONUS: Decile analysis for more granular view
-- ============================================================================

/*
WITH engagement_deciles AS (
    SELECT 
        *,
        NTILE(10) OVER (ORDER BY pages_visited * time_spent) as engagement_decile
    FROM customer_sessions
)
SELECT 
    engagement_decile,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(pages_visited), 2) as avg_pages,
    ROUND(AVG(time_spent), 2) as avg_duration
FROM engagement_deciles
GROUP BY engagement_decile
ORDER BY engagement_decile;
*/

-- ============================================================================
-- BUSINESS INSIGHT:
-- Percentile analysis helps identify:
-- 1. Is there a clear relationship between engagement and conversion?
-- 2. Which engagement level generates the most revenue?
-- 3. Are there diminishing returns at high engagement levels?
-- 4. What's the "minimum viable engagement" for conversion?
-- ============================================================================
