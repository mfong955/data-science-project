-- ============================================================================
-- Query 5: Discount Impact Analysis
-- ============================================================================
-- Purpose: Analyze how discounts affect conversion rates
-- Level: Basic (Level 1)
-- 
-- SQL Concepts Used:
--   CASE WHEN for tiered categorization
--   Custom ORDER BY with CASE - Sorting in non-alphabetical order
--   Business logic in SQL - Translating business questions to queries
-- ============================================================================

SELECT 
    -- Create discount tiers using CASE WHEN
    -- This groups continuous discount values into meaningful business categories
    CASE 
        WHEN discount_applied = 0 THEN 'No Discount'
        WHEN discount_applied <= 10 THEN '1-10%'
        WHEN discount_applied <= 20 THEN '11-20%'
        ELSE '20%+'
    END as discount_tier,
    
    -- Count sessions in each tier
    COUNT(*) as sessions,
    
    -- Conversion rate per tier
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    
    -- Average order value for purchasers in each tier
    ROUND(AVG(CASE WHEN purchase_decision = 1 THEN price END), 2) as avg_order_value,
    
    -- Total revenue per tier
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue,
    
    -- Average discount actually applied (for verification)
    ROUND(AVG(discount_applied), 2) as avg_discount_pct

FROM customer_sessions

GROUP BY discount_tier

-- Custom ORDER BY: We want logical order (No Discount → 1-10% → 11-20% → 20%+)
-- not alphabetical order (which would be: 1-10%, 11-20%, 20%+, No Discount)
ORDER BY 
    CASE discount_tier
        WHEN 'No Discount' THEN 1
        WHEN '1-10%' THEN 2
        WHEN '11-20%' THEN 3
        ELSE 4
    END;

-- ============================================================================
-- BUSINESS INSIGHT:
-- This query helps answer: "Do discounts actually increase conversion?"
-- 
-- Look for:
-- 1. Does conversion_rate increase with higher discount tiers?
-- 2. Does avg_order_value decrease with higher discounts? (price sensitivity)
-- 3. Which tier generates the most total_revenue?
-- 4. Is there a "sweet spot" discount level?
-- ============================================================================
