"""
SQL Database Setup Script
=========================
This script creates a SQLite database from the consumer behavior CSV data.

As per 03_ANALYSIS_PLAN.md (line 482-513):
- Database Location: project/data/processed/ecommerce.db
- SQL Files Location: project/sql/queries/

This script demonstrates:
1. Loading CSV data with pandas
2. Creating a SQLite database
3. Using SQLAlchemy for database connections
4. Verifying the database was created correctly

USAGE:
------
Option 1: Run as script from project root:
    python project/sql/setup_database.py

Option 2: Run in interactive mode (Jupyter/IPython):
    Copy and paste the code, or run:
    %run project/sql/setup_database.py
"""

import pandas as pd
import sqlite3
from pathlib import Path
import os

# ============================================================================
# SETUP PATHS (handles both script and interactive mode)
# ============================================================================


# Method to find project root that works in both script and interactive mode
def find_project_root():
    """
    Find the project root directory.
    Works whether running as a script or in interactive mode.

    The project root is identified by having a 'project' folder with 'data/raw' inside.
    """
    # Try 1: If running as a script, use __file__
    try:
        # __file__ is defined when running as a script
        script_path = Path(__file__).resolve()
        # This file is in: project/sql/setup_database.py
        # Project root is: project/
        project_root = script_path.parent.parent
        if (project_root / "data" / "raw").exists():
            return project_root
    except NameError:
        # __file__ is not defined in interactive mode
        pass

    # Try 2: Use current working directory and search for project structure
    cwd = Path.cwd()

    # Check if we're in the workspace root (data-science-project/)
    if (cwd / "project" / "data" / "raw").exists():
        return cwd / "project"

    # Check if we're already in the project folder
    if (cwd / "data" / "raw").exists():
        return cwd

    # Check if we're in a subfolder of project (e.g., project/sql or project/notebooks)
    for parent in cwd.parents:
        if (parent / "data" / "raw").exists():
            return parent
        if (parent / "project" / "data" / "raw").exists():
            return parent / "project"

    # Fallback: assume we're in workspace root
    raise FileNotFoundError(
        "Could not find project root. Please run from the workspace root directory "
        "(data-science-project/) or from within the project/ folder."
    )


# Find project root
PROJECT_ROOT = find_project_root()

# Define paths for data and database
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "consumer_behavior_dataset.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "ecommerce.db"

# Ensure the processed directory exists
# Path.mkdir() - Creates directory and all parent directories if they don't exist (first use)
# parents=True - Create parent directories as needed
# exist_ok=True - Don't raise error if directory already exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Data source: {DATA_PATH}")
print(f"Database will be created at: {DB_PATH}")

# ============================================================================
# LOAD CSV DATA
# ============================================================================

print("\n=== LOADING CSV DATA ===")

# pd.read_csv() - Reads a CSV file into a DataFrame
# This is the standard way to load tabular data in pandas
df = pd.read_csv(DATA_PATH)

print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
print(f"\nColumn names: {list(df.columns)}")

# ============================================================================
# CREATE SQLITE DATABASE
# ============================================================================

print("\n=== CREATING SQLITE DATABASE ===")

# Method 1: Using pandas to_sql() with sqlite3 connection
# sqlite3.connect() - Creates a connection to a SQLite database (first use)
# If the database file doesn't exist, it will be created
# SQLite is a lightweight, file-based database - perfect for learning and small projects
conn = sqlite3.connect(DB_PATH)

# df.to_sql() - Writes a DataFrame to a SQL database table (first use)
# Parameters:
#   name - Name of the table to create
#   con - Database connection object
#   if_exists='replace' - Drop table if it exists and recreate it
#       Other options: 'fail' (raise error), 'append' (add rows)
#   index=False - Don't write the DataFrame index as a column
df.to_sql("customer_sessions", conn, if_exists="replace", index=False)

print(f"✓ Created table 'customer_sessions' with {len(df):,} rows")

# ============================================================================
# CREATE INDEXES FOR PERFORMANCE
# ============================================================================

print("\n=== CREATING INDEXES ===")

# cursor = conn.cursor() - Creates a cursor object to execute SQL commands (first use)
# A cursor is like a pointer that lets you execute SQL and fetch results
cursor = conn.cursor()

# CREATE INDEX - Creates an index on a column for faster queries (first use)
# Indexes speed up SELECT queries but slow down INSERT/UPDATE operations
# Use indexes on columns you frequently filter or join on

# Index on purchase_decision (our target variable - frequently filtered)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_purchase_decision 
    ON customer_sessions(purchase_decision)
""")

# Index on category (frequently used in GROUP BY)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_category 
    ON customer_sessions(category)
""")

# Index on purchase_date (for time-based queries)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_purchase_date 
    ON customer_sessions(purchase_date)
""")

# conn.commit() - Saves all changes to the database (first use)
# Without commit(), changes are not persisted
conn.commit()

print("✓ Created indexes on: purchase_decision, category, purchase_date")

# ============================================================================
# VERIFY DATABASE
# ============================================================================

print("\n=== VERIFYING DATABASE ===")

# Query to check what tables exist in the database
# sqlite_master is a special table that contains database schema information
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# cursor.fetchall() - Retrieves all rows from the last query (first use)
# Returns a list of tuples
tables = cursor.fetchall()
print(f"Tables in database: {[t[0] for t in tables]}")

# Query to check table structure
# PRAGMA table_info() - SQLite-specific command to get column information
cursor.execute("PRAGMA table_info(customer_sessions);")
columns = cursor.fetchall()
print(f"\nTable structure:")
for col in columns:
    # col format: (cid, name, type, notnull, default_value, pk)
    print(f"  - {col[1]}: {col[2]}")

# Query to count rows
cursor.execute("SELECT COUNT(*) FROM customer_sessions;")

# cursor.fetchone() - Retrieves a single row from the last query (first use)
# Returns a tuple or None if no rows
row_count = cursor.fetchone()[0]
print(f"\nTotal rows in table: {row_count:,}")

# Preview first few rows using pandas
# pd.read_sql() - Executes a SQL query and returns results as DataFrame (first use)
# This is the most common way to query databases in data science workflows
df_preview = pd.read_sql("SELECT * FROM customer_sessions LIMIT 5;", conn)
print(f"\nPreview of data:")
print(df_preview.to_string())

# ============================================================================
# SAMPLE QUERIES TO TEST
# ============================================================================

print("\n=== SAMPLE QUERIES ===")

# Query 1: Overall conversion rate
query1 = """
SELECT 
    COUNT(*) as total_sessions,
    SUM(purchase_decision) as total_purchases,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate_pct
FROM customer_sessions;
"""
result1 = pd.read_sql(query1, conn)
print("\n1. Overall Metrics:")
print(result1.to_string(index=False))

# Query 2: Conversion by category
query2 = """
SELECT 
    category,
    COUNT(*) as sessions,
    ROUND(AVG(purchase_decision) * 100, 2) as conversion_rate
FROM customer_sessions
GROUP BY category
ORDER BY conversion_rate DESC;
"""
result2 = pd.read_sql(query2, conn)
print("\n2. Conversion by Category:")
print(result2.to_string(index=False))

# ============================================================================
# CLEANUP
# ============================================================================

# conn.close() - Closes the database connection (first use)
# Always close connections when done to free up resources
conn.close()

print(f"\n{'=' * 60}")
print(f"✓ Database setup complete!")
print(f"✓ Database saved to: {DB_PATH}")
print(f"✓ You can now run SQL queries against this database")
print(f"{'=' * 60}")
