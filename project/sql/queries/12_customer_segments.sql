-- ============================================================================
-- Query 12: Customer Segments (SQL-based Segmentation)
-- ============================================================================
-- Purpose: Segment customers by behavior patterns using SQL logic
-- Level: Intermediate (Level 2)
-- 
-- SQL Concepts Used:
--   Complex CASE WHEN logic - Multiple conditions for segmentation
--   Business rule implementation - Translating segment definitions to SQL
--   Segment analysis - Comparing metrics across segments
-- ============================================================================

SELECT 
    -- Create customer segments based on behavioral rules
    -- The order of CASE WHEN matters - first match wins
    -- These segments are based on the analysis plan's expected segments:
    --   Power Shoppers: High engagement, high conversion, high value
    --   Window Shoppers: High engagement, low conversion
    --   Quick Deciders: Low engagement, high conversion
    --   Deal Seekers: High discount sensitivity
    CASE 
        -- Power Shoppers: Engaged AND converted (best customers)
        WHEN pages_visited >= 10 AND purchase_decision = 1 THEN 'Power Shoppers'
        
        -- Window Shoppers: Engaged but didn't convert (opportunity)
        WHEN pages_visited >= 10 AND purchase_decision = 0 THEN 'Window Shoppers'
        
        -- Quick Deciders: Low engagement but converted (efficient buyers)
        WHEN pages_visited < 5 AND purchase_decision = 1 THEN 'Quick Deciders'
        
        -- Deal Seekers: High discount users (price sensitive)
        WHEN discount_applied > 15 THEN 'Deal Seekers'
        
        -- Everyone else
        ELSE 'Other'
    END as segment,
    
    -- Segment size
    COUNT(*) as count,
    
    -- Segment percentage of total
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_sessions), 2) as pct_of_total,
    
    -- Conversion rate per segment
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate,
    
    -- Average price interest
    ROUND(AVG(price), 2) as avg_price,
    
    -- Average session duration
    ROUND(AVG(time_spent), 2) as avg_duration,
    
    -- Average pages visited
    ROUND(AVG(pages_visited), 2) as avg_pages,
    
    -- Average discount used
    ROUND(AVG(discount_applied), 2) as avg_discount,
    
    -- Total revenue from segment
    ROUND(SUM(CASE WHEN purchase_decision = 1 THEN price ELSE 0 END), 2) as total_revenue

FROM customer_sessions

GROUP BY segment

ORDER BY conversion_rate DESC;

-- ============================================================================
-- BUSINESS INSIGHTS:
-- 
-- Power Shoppers:
--   - Your best customers - high engagement AND conversion
--   - Strategy: Loyalty programs, early access, premium service
--
-- Window Shoppers:
--   - High engagement but no purchase - biggest opportunity
--   - Strategy: Retargeting, social proof, urgency messaging
--
-- Quick Deciders:
--   - Know what they want, buy fast
--   - Strategy: Streamlined checkout, quick reorder options
--
-- Deal Seekers:
--   - Price sensitive, respond to discounts
--   - Strategy: Discount alerts, bundle deals, flash sales
-- ============================================================================
