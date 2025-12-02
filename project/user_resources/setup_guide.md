# Setup Guide

## 🛠️ Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git installed
- Kaggle account (to download dataset)
- 5GB free disk space

---

## Step 1: Create Project Directory

```bash
# Create main project folder
mkdir ecommerce-analysis
cd ecommerce-analysis

# Create subdirectories
mkdir -p data notebooks src sql_queries dashboard project/plan
```

---

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Option B: Using conda
```bash
conda create -n ecommerce python=3.10
conda activate ecommerce
```

---

## Step 3: Install Required Packages

```bash
# Core data science packages
pip install pandas==2.1.4
pip install numpy==1.26.3
pip install scipy==1.11.4

# Machine learning
pip install scikit-learn==1.4.0

# Visualization
pip install matplotlib==3.8.2
pip install seaborn==0.13.1
pip install plotly==5.18.0

# Jupyter
pip install jupyter==1.0.0
pip install ipykernel==6.28.0

# Database
pip install sqlalchemy==2.0.25

# Optional: Modern alternatives
pip install polars==0.20.3  # Faster than Pandas
pip install duckdb==0.9.2   # Modern SQL engine

# Optional: Dashboard
pip install streamlit==1.30.0

# Save installed packages
pip freeze > requirements.txt
```

---

## Step 4: Download Dataset

### Method 1: Kaggle Website
1. Go to: https://www.kaggle.com/datasets/ziya07/ai-driven-consumer-behavior-dataset
2. Click "Download" button
3. Unzip and move `consumer_behavior_dataset.csv` to `data/` folder

### Method 2: Kaggle API (Faster)
```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials (first time only)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Move downloaded kaggle.json to:
#    - Linux/Mac: ~/.kaggle/kaggle.json
#    - Windows: C:\Users\<username>\.kaggle\kaggle.json

# Download dataset
kaggle datasets download -d ziya07/ai-driven-consumer-behavior-dataset
unzip ai-driven-consumer-behavior-dataset.zip -d data/
```

---

## Step 5: Verify Setup

Create a test notebook to verify everything works:

```bash
# Start Jupyter
jupyter notebook
```

In a new notebook, run:

```python
# Test imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from scipy import stats
import sqlite3
from sqlalchemy import create_engine

# Test data loading
df = pd.read_csv('data/consumer_behavior_dataset.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Columns: {df.columns.tolist()}")

# Test visualization
fig = px.histogram(df, x='purchase_decision', title='Purchase Decision Distribution')
fig.show()
print("✓ All packages working!")
```

---

## Step 6: Initialize Git Repository

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << EOF
# Virtual environment
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Data (don't commit large files)
*.csv
*.xlsx
*.db

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF

# Initial commit
git add .
git commit -m "Initial project setup"
```

---

## Step 7: Project Structure Verification

Your directory should look like this:

```
ecommerce-analysis/
├── data/
│   └── consumer_behavior_dataset.csv
├── notebooks/
│   └── (notebooks will go here)
├── src/
│   └── (Python modules will go here)
├── sql_queries/
│   └── (SQL files will go here)
├── dashboard/
│   └── (Streamlit app will go here)
├── project/
│   └── plan/
│       └── progress.md
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step 8: Create Initial Files

### Create progress.md
```bash
cat > project/plan/progress.md << EOF
# Project Progress Tracker

## Week 1: Foundation & Core Analysis

### Day 1-2: Setup & EDA
- [ ] Environment setup complete
- [ ] Dataset downloaded and explored
- [ ] Initial data quality checks
- [ ] Basic statistics calculated

### Day 3-4: SQL & Metrics
- [ ] SQLite database created
- [ ] Core SQL queries written
- [ ] Key metrics defined
- [ ] Conversion funnel analyzed

### Day 5-7: Segmentation
- [ ] Customer segments identified
- [ ] K-means clustering performed
- [ ] Segment profiles created
- [ ] Visualization completed

## Week 2: Advanced Analysis & Polish

### Day 8-9: Predictive Modeling
- [ ] Features engineered
- [ ] Models trained (Logistic, RF, XGBoost)
- [ ] Model evaluation completed
- [ ] Feature importance analyzed

### Day 10-11: A/B Testing
- [ ] Test scenarios designed
- [ ] Statistical tests performed
- [ ] Effect sizes calculated
- [ ] Results visualized

### Day 12-14: Documentation & Polish
- [ ] Notebooks cleaned and documented
- [ ] README written
- [ ] Resume bullets drafted
- [ ] GitHub repository published

## Resume Updates
- [ ] 3-4 bullets written
- [ ] Talking points prepared
- [ ] Project added to resume

## Next Steps
- [ ] Share on LinkedIn
- [ ] Add to portfolio website
- [ ] Prepare for interviews
EOF
```

### Create README.md template
```bash
cat > README.md << EOF
# E-Commerce Purchase Prediction & Customer Journey Optimization

A comprehensive data science project analyzing consumer behavior patterns, building predictive models, and generating actionable business insights.

## 🎯 Project Overview

[To be filled in after analysis]

## 📊 Key Findings

[To be filled in after analysis]

## 🛠️ Tech Stack

**Languages & Tools:**
- Python 3.10+
- SQL (SQLite/SQLAlchemy)
- Jupyter Notebooks

**Libraries:**
- Data: Pandas, NumPy, Polars
- ML: scikit-learn, SciPy
- Viz: Plotly, Seaborn, Matplotlib
- Stats: SciPy, statsmodels

## 📁 Project Structure

[See directory tree above]

## 🚀 Getting Started

[Instructions to be added]

## 📈 Results

[Analysis results to be added]

## 💼 Skills Demonstrated

[To be filled in]

## 📝 Author

Matthew Fong | [LinkedIn](https://linkedin.com/in/matthew-w-fong) | [GitHub](https://github.com/mfong955)
EOF
```

---

## ✅ Setup Complete Checklist

Before moving to analysis, verify:

- [ ] Python environment activated
- [ ] All packages installed (test imports work)
- [ ] Dataset downloaded to data/ folder
- [ ] Jupyter Notebook launches successfully
- [ ] Git repository initialized
- [ ] project/plan/progress.md created
- [ ] Can load and preview dataset
- [ ] All directories created

---

## 🆘 Troubleshooting

### Issue: ImportError for packages
```bash
# Make sure virtual environment is activated
# Reinstall packages
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found
```bash
# Install kernel
python -m ipykernel install --user --name=ecommerce
```

### Issue: Dataset too large / memory issues
```python
# Load in chunks or use Polars
import polars as pl
df = pl.read_csv('data/consumer_behavior_dataset.csv')
```

### Issue: Kaggle API authentication
```bash
# Check credentials location
ls ~/.kaggle/kaggle.json  # Mac/Linux
dir C:\Users\<username>\.kaggle\kaggle.json  # Windows

# Fix permissions (Mac/Linux)
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📚 Additional Resources

- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Plotly Python: https://plotly.com/python/
- SQL Tutorial: https://www.sqltutorial.org/

---

**Next Step:** Read 02_DATASET_INFO.md to understand your data!