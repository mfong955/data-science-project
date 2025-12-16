-- ============================================================================
-- Query 4: Demographics Breakdown
-- ============================================================================
-- Purpose: Analyze conversion by age group and gender
-- Level: Basic (Level 1)
-- 
-- SQL Concepts Used:
--   CASE WHEN with multiple conditions - Creating age buckets
--   Multiple GROUP BY columns - Grouping by two dimensions
--   Derived columns - Creating new columns from existing data
-- ============================================================================

SELECT 
    -- Create age groups using CASE WHEN
    -- This is called "bucketing" or "binning" - converting continuous data to categories
    -- CASE evaluates conditions in order and returns the first match
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    
    -- Gender from the original data
    gender,
    
    -- Metrics per age_group + gender combination
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    ROUND(AVG(price), 2) as avg_price,
    ROUND(AVG(time_spent), 2) as avg_session_duration

FROM customer_sessions

-- GROUP BY multiple columns creates groups for each unique combination
-- e.g., "18-24 + Male", "18-24 + Female", "25-34 + Male", etc.
GROUP BY age_group, gender

-- ORDER BY multiple columns: first by age_group, then by gender within each age_group
ORDER BY age_group, gender;

-- ============================================================================
-- BONUS: Just by age group (without gender split)
-- ============================================================================

/*
SELECT 
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END) as total_revenue
FROM customer_sessions
GROUP BY age_group
ORDER BY 
    -- Custom sort order for age groups (not alphabetical)
    CASE age_group
        WHEN '18-24' THEN 1
        WHEN '25-34' THEN 2
        WHEN '35-44' THEN 3
        WHEN '45-54' THEN 4
        ELSE 5
    END;
*/
