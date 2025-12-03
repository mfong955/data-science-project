# Setup Guide

## 🛠️ Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git installed
- 5GB free disk space

**Note:** Dataset already downloaded to `project/data/raw/consumer_behavior_dataset.csv`

---

## Step 1: Navigate to Project Directory

```bash
cd data-science-project
```

The project structure is already set up:
```
project/
├── data/
│   └── raw/
│       └── consumer_behavior_dataset.csv  # Already downloaded!
├── notebooks/
│   ├── exploratory/
│   ├── modeling/
│   └── reports/
├── src/
├── sql/
├── models/
├── experiments/
├── visualizations/
└── ...
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

## Step 4: Verify Setup

Create a test notebook to verify everything works:

```bash
# Start Jupyter
jupyter notebook
```

In a new notebook (save to `project/notebooks/exploratory/`), run:

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

# Test data loading - using our project structure
df = pd.read_csv('../../data/raw/consumer_behavior_dataset.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Columns: {df.columns.tolist()}")

# Test visualization
fig = px.histogram(df, x='purchase_decision', title='Purchase Decision Distribution')
fig.show()
print("✓ All packages working!")
```

---

## Step 5: Initialize Git Repository (if not already done)

```bash
# Initialize git (if needed)
git init

# The .gitignore is already created at project/.gitignore
# It excludes:
# - Data files (*.csv)
# - Model files
# - Virtual environments
# - Jupyter checkpoints
# - IDE files
```

---

## Step 6: Project Structure Verification

Your directory should look like this:

```
data-science-project/
├── README.md
├── ai_system/
│   ├── personas/
│   │   └── data_science_personas.md
│   ├── memory/
│   │   └── user_profile.md
│   ├── rules/
│   │   └── aiworkspace.md
│   └── templates/
├── project/
│   ├── data/
│   │   ├── raw/
│   │   │   └── consumer_behavior_dataset.csv  ✓ Already here!
│   │   ├── processed/
│   │   └── external/
│   ├── notebooks/
│   │   ├── exploratory/    # Put EDA notebooks here
│   │   ├── modeling/       # Put model notebooks here
│   │   └── reports/        # Put final notebooks here
│   ├── src/
│   ├── sql/
│   │   └── queries/        # Put SQL files here
│   ├── models/
│   ├── experiments/
│   ├── visualizations/
│   ├── plan/
│   │   ├── goals.md
│   │   └── progress.md     # Track progress here!
│   └── user_resources/     # Planning docs (you are here)
├── venv/                   # Your virtual environment
└── requirements.txt
```

---

## ✅ Setup Complete Checklist

Before moving to analysis, verify:

- [x] Dataset downloaded to `project/data/raw/consumer_behavior_dataset.csv`
- [ ] Python environment activated
- [ ] All packages installed (test imports work)
- [ ] Jupyter Notebook launches successfully
- [ ] Can load and preview dataset
- [ ] Git repository initialized (optional)

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

### Issue: Dataset path not found
```python
# If running from project/notebooks/exploratory/:
df = pd.read_csv('../../data/raw/consumer_behavior_dataset.csv')

# If running from project root:
df = pd.read_csv('project/data/raw/consumer_behavior_dataset.csv')

# If running from data-science-project root:
df = pd.read_csv('project/data/raw/consumer_behavior_dataset.csv')
```

### Issue: Dataset too large / memory issues
```python
# Load in chunks or use Polars
import polars as pl
df = pl.read_csv('../../data/raw/consumer_behavior_dataset.csv')
```

---

## 📚 Additional Resources

- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Plotly Python: https://plotly.com/python/
- SQL Tutorial: https://www.sqltutorial.org/

---

## 📁 Where to Put Your Work

| File Type | Location |
|-----------|----------|
| EDA notebooks | `project/notebooks/exploratory/` |
| Model notebooks | `project/notebooks/modeling/` |
| Final notebooks | `project/notebooks/reports/` |
| SQL queries | `project/sql/queries/` |
| Trained models | `project/models/trained/` |
| Figures | `project/visualizations/figures/` |
| Processed data | `project/data/processed/` |

---

**Next Step:** Read 02_DATASET_INFO.md to understand your data!