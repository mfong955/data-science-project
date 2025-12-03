# AI Assistant Prompt

## 🤖 How to Use This Document

Copy and paste the relevant sections below when starting a new AI conversation about this project. This ensures the AI has proper context about your project structure, goals, and preferences.

---

## 📋 Full Project Context Prompt

Use this when starting a new conversation or when the AI needs complete context:

```
I'm working on a data science portfolio project for job applications, specifically targeting a Product Data Scientist role at OpenAI.

## Project Overview
- **Project:** E-Commerce Purchase Prediction & Customer Journey Optimization
- **Dataset:** AI-Driven Consumer Behavior Dataset (5,000 customer sessions)
- **Goal:** Build purchase prediction models and derive business insights
- **Timeline:** 2 weeks

## Project Structure
```
project/
├── data/
│   ├── raw/consumer_behavior_dataset.csv    # Original dataset (downloaded)
│   └── processed/                           # Cleaned data, SQLite database
├── notebooks/                               # Jupyter notebooks (01-05)
├── src/                                     # Python modules
│   ├── data/                                # Data loading utilities
│   ├── features/                            # Feature engineering
│   ├── models/                              # Model implementations
│   └── visualization/                       # Plotting utilities
├── models/                                  # Saved model files
├── experiments/                             # MLflow experiment tracking
├── visualizations/                          # Saved plots and figures
├── sql/queries/                             # SQL query files
├── tests/                                   # Unit tests
├── configs/                                 # Configuration files
├── docs/                                    # Documentation
├── plan/                                    # Goals and progress tracking
│   ├── goals.md
│   └── progress.md
├── context/                                 # Session notes
│   └── session_notes.md
└── user_resources/                          # Reference materials (00-07)
```

## Key Files
- **Dataset:** `project/data/raw/consumer_behavior_dataset.csv`
- **Goals:** `project/plan/goals.md`
- **Progress:** `project/plan/progress.md`
- **User Profile:** `ai_system/memory/user_profile.md`
- **Personas:** `ai_system/personas/data_science_personas.md`

## Technical Stack
- Python, SQL, PyTorch, TensorFlow, JAX, scikit-learn
- pandas, NumPy, Matplotlib, Seaborn
- Jupyter notebooks

## My Background
- PhD in Physics with statistical modeling experience
- Self-taught in ML, built custom algorithms for research
- Learning industry-standard data science tools
- AWS experience, targeting OpenAI Product Data Scientist role

## Current Status
[Describe what you're currently working on]

## What I Need Help With
[Describe your specific request]
```

---

## 🎯 Quick Context Prompts

### For Notebook Development
```
I'm working on a data science portfolio project.

Project: E-Commerce Purchase Prediction
Dataset: project/data/raw/consumer_behavior_dataset.csv (5,000 sessions)
Current notebook: project/notebooks/[NOTEBOOK_NAME]

The dataset has these columns:
- user_id, product_id (identifiers)
- category, price, discount_applied, payment_method (product/transaction)
- purchase_date (date of session)
- pages_visited, time_spent, add_to_cart, abandoned_cart (behavioral)
- rating, review_text, sentiment_score (review features)
- age, gender, income_level, location (demographics)
- purchase_decision (target: 0/1)

I need help with: [YOUR REQUEST]
```

### For SQL Queries
```
I'm practicing SQL for a data science portfolio project.

Database: project/data/processed/ecommerce.db
Table: customer_sessions
Query files: project/sql/queries/

The table has columns for customer demographics, session behavior, and purchase decisions.

I need help writing a query to: [YOUR REQUEST]
```

### For Model Development
```
I'm building ML models for purchase prediction.

Data location: project/data/raw/consumer_behavior_dataset.csv
Models save to: project/models/
Target: purchase_decision (binary classification)

I'm implementing in [PyTorch/TensorFlow/JAX/sklearn].

I need help with: [YOUR REQUEST]
```

### For Code Review
```
I'm working on a data science portfolio project for job applications.

Please review my code for:
- Best practices
- Efficiency improvements
- Documentation quality
- Production readiness

Code location: [FILE PATH]

[PASTE CODE OR DESCRIBE WHAT TO REVIEW]
```

---

## 🧑‍💼 Persona-Specific Prompts

### Data Engineer Persona
```
Act as a Data Engineer helping me with data pipeline development.

Focus on:
- Data loading and validation
- SQL database design
- ETL processes
- Data quality checks

Project context:
- Dataset: project/data/raw/consumer_behavior_dataset.csv
- Database: project/data/processed/ecommerce.db
- SQL queries: project/sql/queries/
```

### ML Engineer (PyTorch) Persona
```
Act as an ML Engineer specializing in PyTorch.

Focus on:
- Custom Dataset and DataLoader classes
- Neural network architecture design
- Training loop optimization
- Model checkpointing

Project context:
- Data: project/data/raw/consumer_behavior_dataset.csv
- Models: project/models/
- Task: Binary classification (purchase prediction)
```

### ML Engineer (TensorFlow) Persona
```
Act as an ML Engineer specializing in TensorFlow/Keras.

Focus on:
- Keras Sequential and Functional API
- Callbacks (EarlyStopping, ModelCheckpoint, etc.)
- TensorBoard integration
- Model serving preparation

Project context:
- Data: project/data/raw/consumer_behavior_dataset.csv
- Models: project/models/
- Task: Binary classification (purchase prediction)
```

### ML Engineer (JAX) Persona
```
Act as an ML Engineer specializing in JAX/Flax.

Focus on:
- Functional programming patterns
- JIT compilation optimization
- Flax module design
- Optax optimizers

Project context:
- Data: project/data/raw/consumer_behavior_dataset.csv
- Models: project/models/
- Task: Binary classification (purchase prediction)
```

### Data Scientist (Statistics) Persona
```
Act as a Data Scientist with strong statistics background.

Focus on:
- Exploratory data analysis
- Statistical hypothesis testing
- Feature selection methods
- Model evaluation and comparison

Project context:
- Dataset: project/data/raw/consumer_behavior_dataset.csv
- Notebooks: project/notebooks/
- Target: purchase_decision (binary)
```

### Data Visualization Specialist Persona
```
Act as a Data Visualization Specialist.

Focus on:
- Clear, publication-quality plots
- Interactive visualizations
- Dashboard design
- Storytelling with data

Project context:
- Data: project/data/raw/consumer_behavior_dataset.csv
- Save plots to: project/visualizations/
- Style: Professional, suitable for portfolio
```

---

## 📝 Session Continuation Prompt

Use this when continuing work from a previous session:

```
I'm continuing work on my data science portfolio project.

## Previous Session Summary
[Copy relevant notes from project/context/session_notes.md]

## Current Progress
[Check project/plan/progress.md for status]

## Today's Goals
[What you want to accomplish]

## Where I Left Off
[Specific file/notebook and what was being worked on]
```

---

## ⚠️ Important Reminders for AI

Include these when needed:

### For Code Generation
```
Please ensure:
- Code is complete and runnable (no placeholders)
- Includes proper imports
- Has docstrings and comments
- Follows PEP 8 style
- Uses relative paths from notebooks/ folder
```

### For File Paths
```
Important path context:
- Notebooks are in: project/notebooks/
- Raw data is in: project/data/raw/
- Use pathlib with __file__ for reliable paths:
  PROJECT_ROOT = Path(__file__).parent.parent.parent
  DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'consumer_behavior_dataset.csv'
- Save models to: project/models/
- Save plots to: project/visualizations/
```

### For Business Context
```
Remember this is a portfolio project for:
- Target role: Product Data Scientist at OpenAI
- Focus: Product analytics, experimentation, ML
- Audience: Technical recruiters and hiring managers
- Goal: Demonstrate both technical skills and business thinking
```

---

## 🔄 End of Session Prompt

Use this to wrap up a session:

```
Please help me document this session:

1. Summarize what we accomplished
2. List any files created or modified
3. Note any issues or blockers encountered
4. Suggest next steps

I'll save this to: project/context/session_notes.md
```

---

## 📁 File Reference

| Resource | Location |
|----------|----------|
| Dataset | `project/data/raw/consumer_behavior_dataset.csv` |
| Notebooks | `project/notebooks/` |
| SQL Queries | `project/sql/queries/` |
| Models | `project/models/` |
| Visualizations | `project/visualizations/` |
| Goals | `project/plan/goals.md` |
| Progress | `project/plan/progress.md` |
| Session Notes | `project/context/session_notes.md` |
| User Profile | `ai_system/memory/user_profile.md` |
| Personas | `ai_system/personas/data_science_personas.md` |
| User Resources | `project/user_resources/00-07_*.md` |

---

## ✅ Checklist Before Starting AI Session

- [ ] Know which notebook/file you're working on
- [ ] Have specific goals for the session
- [ ] Check progress.md for current status
- [ ] Prepare any error messages or code snippets to share
- [ ] Know which persona would be most helpful

---

**Tip:** Save successful prompts and AI responses in `project/context/session_notes.md` for future reference!