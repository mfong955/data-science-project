-- ============================================================================
-- Query 16: A/B Test Comparison (Simulated)
-- ============================================================================
-- Purpose: Compare high discount vs low discount groups as if it were an A/B test
-- Level: Advanced (Level 3)
-- 
-- SQL Concepts Used:
--   Simulated A/B test analysis - Treating existing data as test groups
--   Statistical comparison setup - Preparing data for significance testing
--   Multiple CTEs for clean analysis flow
--   COALESCE() - Replace NULL with default value
-- 
-- NOTE: This is a simulated A/B test using observational data.
-- In a real A/B test, users would be randomly assigned to groups.
-- ============================================================================

-- CTE 1: Assign users to "test groups" based on discount level
WITH discount_groups AS (
    SELECT 
        -- Create two groups: High Discount (treatment) vs Low Discount (control)
        CASE 
            WHEN discount_applied >= 15 THEN 'Treatment (High Discount)'
            ELSE 'Control (Low Discount)'
        END as test_group,
        
        -- Keep relevant columns for analysis
        purchase_decision,
        price,
        discount_applied,
        pages_visited,
        time_spent
    FROM customer_sessions
),

-- CTE 2: Calculate statistics for each group
group_stats AS (
    SELECT 
        test_group,
        
        -- Sample size (n)
        COUNT(*) as n,
        
        -- Number of conversions
        SUM(purchase_decision) as conversions,
        
        -- Conversion rate (proportion)
        AVG(purchase_decision) as conv_rate,
        
        -- Standard error for proportion: sqrt(p * (1-p) / n)
        -- This is used for confidence interval calculation
        SQRT(AVG(purchase_decision) * (1 - AVG(purchase_decision)) / COUNT(*)) as std_error,
        
        -- Average order value (for purchasers only)
        AVG(CASE WHEN purchase_decision = 1 THEN price END) as avg_order_value,
        
        -- Total revenue
        SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END) as total_revenue,
        
        -- Average discount in group
        AVG(discount_applied) as avg_discount
    FROM discount_groups
    GROUP BY test_group
)

-- Final output with comparison metrics
SELECT 
    test_group,
    n,
    conversions,
    
    -- Conversion rate as percentage
    ROUND(conv_rate * 100, 2) as conversion_rate_pct,
    
    -- Standard error (for confidence intervals)
    ROUND(std_error * 100, 4) as std_error_pct,
    
    -- 95% Confidence Interval: rate ± 1.96 * std_error
    ROUND((conv_rate - 1.96 * std_error) * 100, 2) as ci_lower_95,
    ROUND((conv_rate + 1.96 * std_error) * 100, 2) as ci_upper_95,
    
    -- Revenue metrics
    ROUND(COALESCE(avg_order_value, 0), 2) as avg_order_value,
    ROUND(total_revenue, 2) as total_revenue,
    
    -- Average discount in group
    ROUND(avg_discount, 2) as avg_discount_pct

FROM group_stats
ORDER BY test_group;

-- ============================================================================
-- CALCULATING STATISTICAL SIGNIFICANCE (conceptual)
-- ============================================================================
-- 
-- To determine if the difference is statistically significant:
-- 
-- 1. Calculate the difference in conversion rates:
--    diff = treatment_rate - control_rate
-- 
-- 2. Calculate pooled standard error:
--    pooled_se = sqrt(se_treatment² + se_control²)
-- 
-- 3. Calculate z-score:
--    z = diff / pooled_se
-- 
-- 4. If |z| > 1.96, the difference is significant at 95% confidence
-- 
-- In Python, you would use:
--    from scipy.stats import proportions_ztest
--    z_stat, p_value = proportions_ztest([conv_treatment, conv_control], 
--                                         [n_treatment, n_control])
-- ============================================================================

-- ============================================================================
-- BONUS: Effect size calculation
-- ============================================================================

/*
WITH group_comparison AS (
    SELECT 
        MAX(CASE WHEN discount_applied >= 15 THEN AVG(purchase_decision) END) as treatment_rate,
        MAX(CASE WHEN discount_applied < 15 THEN AVG(purchase_decision) END) as control_rate
    FROM customer_sessions
)
SELECT 
    ROUND(treatment_rate * 100, 2) as treatment_conversion_pct,
    ROUND(control_rate * 100, 2) as control_conversion_pct,
    ROUND((treatment_rate - control_rate) * 100, 2) as absolute_lift_pct_points,
    ROUND((treatment_rate - control_rate) / control_rate * 100, 2) as relative_lift_pct
FROM group_comparison;
*/

-- ============================================================================
-- BUSINESS INSIGHT:
-- 
-- This analysis helps answer:
-- 1. Does offering higher discounts increase conversion?
-- 2. What's the magnitude of the effect (lift)?
-- 3. Is the effect statistically significant?
-- 4. What's the trade-off between higher conversion and lower AOV?
-- 5. Is the revenue impact positive or negative overall?
-- ============================================================================
