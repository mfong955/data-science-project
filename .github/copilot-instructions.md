# AI Coding Agent Instructions

## Project Overview
This is a portfolio data science project analyzing e-commerce consumer behavior for predictive modeling and business insights. Uses multi-agent AI workflow with specialized personas.

## Architecture
- **Data Flow**: Raw CSV → SQLite DB → Pandas analysis → ML models → visualizations
- **AI System**: Personas in `ai_system/personas/` handle different expertise areas
- **Structure**: Standard DS layout with `project/` containing all code/data

## Session Initialization
At start of each session, read these files in order:
1. `ai_system/personas/data_science_personas.md` - Available expert roles
2. `ai_system/memory/user_profile.md` - User preferences
3. `project/plan/goals.md` - Project objectives
4. `project/plan/progress.md` - Current status
5. `project/context/session_notes.md` - Session context

## Key Conventions
- **Personas**: Use appropriate persona (Data Engineer, ML Engineer, etc.) based on task
- **Paths**: All relative to `project/` (e.g., `data/raw/consumer_behavior_dataset.csv`)
- **Database**: SQLite at `data/processed/ecommerce.db` created via `sql/setup_database.py`
- **Notebooks**: Develop in order (01_EDA → 02_segmentation → 03_modeling → etc.)

## Dataset Schema
19 columns, 5K rows. Key: `user_id`, `product_id`, `category`, `price`, `purchase_decision` (target).

## Critical Workflows
- **Setup**: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
- **DB Setup**: `python project/sql/setup_database.py` (creates SQLite from CSV)
- **Query Execution**: `python project/sql/run_queries.py` (runs SQL files in `sql/queries/`)
- **Model Training**: Save to `models/` with experiment tracking in `experiments/`

## Integration Points
- **External Data**: Kaggle dataset download required
- **SQL Queries**: Pre-written in `sql/queries/` (01_overall_metrics.sql, etc.)
- **AI Memory**: Update `project/context/session_notes.md` with progress
- **Progress Tracking**: Update `project/plan/progress.md` after major tasks

## Code Patterns
- **Imports**: `import pandas as pd; import sqlite3; from pathlib import Path`
- **Path Handling**: Use `Path(__file__).parent.parent / "data" / "raw"` for robustness
- **Persona Indication**: Prefix responses with `*[Persona]*` when assuming role
- **Error Handling**: Basic try/except for file operations, data validation

## Dependencies
Core: pandas, numpy, scikit-learn, matplotlib, seaborn, sqlalchemy, jupyter
Optional: plotly, streamlit, polars, duckdb

Reference: `project/user_resources/07_AI_ASSISTANT_PROMPT.md` for full context prompts.</content>
<parameter name="filePath">c:\Users\matth\OneDrive\Documents\workplace\mwfong\data-science-project\.github\copilot-instructions.md