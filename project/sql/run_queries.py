"""
SQL Query Runner Script
=======================
This script demonstrates how to run SQL queries from Python and display results.

It reads SQL files from project/sql/queries/ and executes them against the
SQLite database at project/data/processed/ecommerce.db

Run this after setup_database.py to see query results.

USAGE:
------
Option 1: Run as script from project root:
    python project/sql/run_queries.py

Option 2: Run in interactive mode (Jupyter/IPython):
    Copy and paste the code, or run:
    %run project/sql/run_queries.py
"""

import pandas as pd
import sqlite3
from pathlib import Path

# ============================================================================
# SETUP PATHS (handles both script and interactive mode)
# ============================================================================


def find_project_root():
    """
    Find the project root directory.
    Works whether running as a script or in interactive mode.

    The project root is identified by having a 'project' folder with 'data/raw' inside.
    """
    # Try 1: If running as a script, use __file__
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        if (project_root / "data" / "raw").exists():
            return project_root
    except NameError:
        pass

    # Try 2: Use current working directory and search for project structure
    cwd = Path.cwd()

    # Check if we're in the workspace root (data-science-project/)
    if (cwd / "project" / "data" / "raw").exists():
        return cwd / "project"

    # Check if we're already in the project folder
    if (cwd / "data" / "raw").exists():
        return cwd

    # Check if we're in a subfolder of project
    for parent in cwd.parents:
        if (parent / "data" / "raw").exists():
            return parent
        if (parent / "project" / "data" / "raw").exists():
            return parent / "project"

    raise FileNotFoundError(
        "Could not find project root. Please run from the workspace root directory."
    )


# Find project root
PROJECT_ROOT = find_project_root()
DB_PATH = PROJECT_ROOT / "data" / "processed" / "ecommerce.db"
QUERIES_PATH = PROJECT_ROOT / "sql" / "queries"

# Check if database exists
if not DB_PATH.exists():
    print(f"❌ Database not found at: {DB_PATH}")
    print("Please run setup_database.py first to create the database.")
    exit(1)

print(f"Database: {DB_PATH}")
print(f"Queries folder: {QUERIES_PATH}")

# ============================================================================
# HELPER FUNCTION TO RUN QUERIES
# ============================================================================


def run_query_from_file(conn, sql_file_path, show_query=True):
    """
    Read a SQL file and execute it, returning results as a DataFrame.

    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection object
    sql_file_path : Path
        Path to the .sql file
    show_query : bool
        Whether to print the SQL query before results

    Returns:
    --------
    pd.DataFrame
        Query results as a pandas DataFrame
    """
    # Read the SQL file
    # Path.read_text() - Reads entire file contents as a string (first use)
    sql_query = sql_file_path.read_text()

    # Remove comment blocks for cleaner execution
    # We keep the query as-is since SQLite handles comments

    if show_query:
        # Print first few lines of the query (skip long comments)
        lines = sql_query.strip().split("\n")
        # Find first non-comment line
        for i, line in enumerate(lines):
            if not line.strip().startswith("--") and line.strip():
                break
        print(f"\n{'=' * 60}")
        print(f"Query: {sql_file_path.name}")
        print(f"{'=' * 60}")

    # Execute query and return results
    # pd.read_sql() handles the query execution and returns a DataFrame
    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def run_inline_query(conn, query, description=""):
    """
    Execute an inline SQL query and return results.

    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection object
    query : str
        SQL query string
    description : str
        Optional description of what the query does

    Returns:
    --------
    pd.DataFrame
        Query results
    """
    if description:
        print(f"\n{description}")
        print("-" * len(description))

    return pd.read_sql(query, conn)


# ============================================================================
# CONNECT TO DATABASE
# ============================================================================

print("\n" + "=" * 60)
print("CONNECTING TO DATABASE")
print("=" * 60)

# sqlite3.connect() creates a connection to the database
conn = sqlite3.connect(DB_PATH)
print(f"✓ Connected to database")

# Quick verification
row_count = pd.read_sql("SELECT COUNT(*) as count FROM customer_sessions", conn)
print(f"✓ Table has {row_count['count'].iloc[0]:,} rows")

# ============================================================================
# RUN LEVEL 1: BASIC QUERIES
# ============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: BASIC QUERIES")
print("=" * 60)

# Query 1: Overall Metrics
query1_path = QUERIES_PATH / "01_overall_metrics.sql"
if query1_path.exists():
    df1 = run_query_from_file(conn, query1_path)
    if df1 is not None:
        print("\nResults:")
        print(df1.to_string(index=False))

# Query 2: Conversion by Category
query2_path = QUERIES_PATH / "02_conversion_by_category.sql"
if query2_path.exists():
    df2 = run_query_from_file(conn, query2_path)
    if df2 is not None:
        print("\nResults:")
        print(df2.to_string(index=False))

# Query 3: Cart Abandonment
query3_path = QUERIES_PATH / "03_cart_abandonment.sql"
if query3_path.exists():
    df3 = run_query_from_file(conn, query3_path)
    if df3 is not None:
        print("\nResults:")
        print(df3.to_string(index=False))

# Query 4: Demographics
query4_path = QUERIES_PATH / "04_demographics.sql"
if query4_path.exists():
    df4 = run_query_from_file(conn, query4_path)
    if df4 is not None:
        print("\nResults:")
        print(df4.to_string(index=False))

# Query 5: Discount Impact
query5_path = QUERIES_PATH / "05_discount_impact.sql"
if query5_path.exists():
    df5 = run_query_from_file(conn, query5_path)
    if df5 is not None:
        print("\nResults:")
        print(df5.to_string(index=False))

# ============================================================================
# RUN LEVEL 2: INTERMEDIATE QUERIES
# ============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: INTERMEDIATE QUERIES")
print("=" * 60)

# Query 8: Engagement Funnel
query8_path = QUERIES_PATH / "08_engagement_funnel.sql"
if query8_path.exists():
    df8 = run_query_from_file(conn, query8_path)
    if df8 is not None:
        print("\nResults:")
        print(df8.to_string(index=False))

# Query 12: Customer Segments
query12_path = QUERIES_PATH / "12_customer_segments.sql"
if query12_path.exists():
    df12 = run_query_from_file(conn, query12_path)
    if df12 is not None:
        print("\nResults:")
        print(df12.to_string(index=False))

# ============================================================================
# RUN LEVEL 3: ADVANCED QUERIES
# ============================================================================

print("\n" + "=" * 60)
print("LEVEL 3: ADVANCED QUERIES")
print("=" * 60)

# Query 13: Percentile Analysis
query13_path = QUERIES_PATH / "13_percentile_analysis.sql"
if query13_path.exists():
    df13 = run_query_from_file(conn, query13_path)
    if df13 is not None:
        print("\nResults:")
        print(df13.to_string(index=False))

# Query 15: Comprehensive Analysis
query15_path = QUERIES_PATH / "15_comprehensive_analysis.sql"
if query15_path.exists():
    df15 = run_query_from_file(conn, query15_path)
    if df15 is not None:
        print("\nResults:")
        print(df15.to_string(index=False))

# Query 16: A/B Test Comparison
query16_path = QUERIES_PATH / "16_ab_test_comparison.sql"
if query16_path.exists():
    df16 = run_query_from_file(conn, query16_path)
    if df16 is not None:
        print("\nResults:")
        print(df16.to_string(index=False))

# ============================================================================
# BONUS: INLINE QUERY EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("BONUS: INLINE QUERY EXAMPLES")
print("=" * 60)

# Example 1: Quick count by category
df_quick = run_inline_query(
    conn,
    """
    SELECT category, COUNT(*) as count 
    FROM customer_sessions 
    GROUP BY category 
    ORDER BY count DESC
    """,
    "Quick count by category:",
)
print(df_quick.to_string(index=False))

# Example 2: Top 5 highest converting price points
df_price = run_inline_query(
    conn,
    """
    SELECT 
        ROUND(price, -1) as price_bucket,  -- Round to nearest 10
        COUNT(*) as sessions,
        ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate
    FROM customer_sessions
    GROUP BY price_bucket
    HAVING COUNT(*) >= 50  -- Only buckets with enough data
    ORDER BY conversion_rate DESC
    LIMIT 5
    """,
    "Top 5 price points by conversion rate (min 50 sessions):",
)
print(df_price.to_string(index=False))

# ============================================================================
# CLEANUP
# ============================================================================

# Always close the connection when done
conn.close()
print("\n" + "=" * 60)
print("✓ All queries completed successfully!")
print("✓ Database connection closed")
print("=" * 60)
