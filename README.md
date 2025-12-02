# Data Science Portfolio Project

A lightweight, intelligent workspace for AI-assisted data science project development. This template is designed for building portfolio projects that demonstrate proficiency in modern data science tools including JAX, TensorFlow, PyTorch, SQL, and industry best practices.

---

## Quick Start

### For Users

1. **Start Your Project**: Open this workspace in VSCode
2. **Begin Working**: Start chatting with your AI assistant and ask it to read this file
3. **AI Automatically**: Reads context files, personas, and understands your project
4. **Work Naturally**: AI assumes appropriate persona based on task
5. **Save Session**: Say "save session" - AI immediately saves all progress

**That's it!** The system handles everything automatically.

---

## Project Goals

This workspace is designed to help you:
- **Learn industry data science tools**: JAX, TensorFlow, PyTorch, SQL
- **Build portfolio projects**: Using public datasets for GitHub showcase
- **Apply scientific background**: Bridge research experience with industry practices
- **Follow best practices**: Project structure, reproducibility, documentation

---

## How It Works

### For AI

The AI reads these files at session start:

1. **Personas** (Select appropriate expert role):
   - [`ai_system/personas/data_science_personas.md`](ai_system/personas/data_science_personas.md) - **START HERE**

2. **Rules** (Understand workspace behavior):
   - [`ai_system/rules/aiworkspace.md`](ai_system/rules/aiworkspace.md)

3. **User Understanding** (Communication style):
   - [`ai_system/memory/user_profile.md`](ai_system/memory/user_profile.md)

4. **Project Context**:
   - [`project/plan/goals.md`](project/plan/goals.md) - Project objectives
   - [`project/plan/progress.md`](project/plan/progress.md) - Current status
   - [`project/context/session_notes.md`](project/context/session_notes.md) - Last session context
   - [`project/history/decisions.md`](project/history/decisions.md) - Recent decisions

### For You
- **Edit any file** to update preferences or project info
- **AI sees changes** automatically at next session
- **Full transparency** - see exactly what AI knows
- **Full control** - edit files anytime

---

## AI Personas

This project uses specialized AI personas for different tasks. See [`ai_system/personas/data_science_personas.md`](ai_system/personas/data_science_personas.md) for full details.

### Quick Reference

| Persona | Slug | Primary Focus |
|---------|------|---------------|
| Data Engineer | `data-engineer` | SQL, pipelines, data infrastructure |
| ML Engineer (PyTorch) | `ml-engineer-pytorch` | Deep learning, research flexibility |
| ML Engineer (TensorFlow) | `ml-engineer-tensorflow` | Production ML, deployment |
| ML Engineer (JAX) | `ml-engineer-jax` | High-performance, functional ML |
| Data Scientist (Stats) | `data-scientist-stats` | Statistical modeling, inference |
| MLOps Engineer | `mlops-engineer` | Experiment tracking, deployment |
| Data Viz Specialist | `data-viz-specialist` | Visualization, dashboards |
| Project Architect | `project-architect` | Structure, standards, planning |

### Invoking a Persona

- **Explicit**: "As the Data Engineer, help me design a schema for..."
- **Task-based**: AI automatically selects based on task type
- **Multi-persona**: "I need the Data Engineer and MLOps Engineer to collaborate on..."

---

## Repository Structure

```
your-project/
├── README.md                              # This file
├── ai_system/
│   ├── personas/
│   │   └── data_science_personas.md       # AI personas for multi-agent workflow
│   ├── memory/
│   │   └── user_profile.md                # Your communication style & preferences
│   ├── rules/
│   │   └── aiworkspace.md                 # AI's instruction file
│   └── templates/
│       ├── data_science_project_template.md  # Data science project structure
│       ├── app_project_template.md        # Template for app projects
│       └── story_project_template.md      # Template for story projects
└── project/
    ├── plan/
    │   ├── goals.md                       # What you want to achieve
    │   └── progress.md                    # Current status & next steps
    ├── context/
    │   └── session_notes.md               # Current session working memory
    ├── history/
    │   └── decisions.md                   # Decision log & session summaries
    ├── user_resources/                    # Your reference materials
    ├── data/                              # Data files
    │   ├── raw/                           # Original, immutable data
    │   ├── processed/                     # Cleaned, transformed data
    │   ├── interim/                       # Intermediate transformations
    │   └── external/                      # External data sources
    ├── notebooks/                         # Jupyter notebooks
    │   ├── exploratory/                   # EDA and experimentation
    │   ├── modeling/                      # Model development
    │   └── reports/                       # Final analysis notebooks
    ├── src/                               # Production code
    │   ├── data/                          # Data loading and processing
    │   ├── features/                      # Feature engineering
    │   ├── models/                        # Model definitions
    │   ├── training/                      # Training scripts
    │   ├── evaluation/                    # Evaluation metrics
    │   └── utils/                         # Utility functions
    ├── models/                            # Model artifacts
    │   ├── trained/                       # Saved models
    │   ├── checkpoints/                   # Training checkpoints
    │   └── configs/                       # Model configurations
    ├── experiments/                       # Experiment tracking
    │   ├── logs/                          # Training logs
    │   └── results/                       # Experiment results
    ├── visualizations/                    # Visualization outputs
    │   ├── figures/                       # Static figures
    │   └── dashboards/                    # Dashboard code
    ├── sql/                               # Database files
    │   ├── queries/                       # SQL queries
    │   ├── schemas/                       # Schema definitions
    │   └── migrations/                    # Migration scripts
    ├── tests/                             # Test files
    │   ├── unit/                          # Unit tests
    │   ├── integration/                   # Integration tests
    │   └── data_validation/               # Data quality tests
    ├── docs/                              # Documentation
    │   ├── guides/                        # How-to guides
    │   ├── api/                           # API documentation
    │   └── research/                      # Research notes
    └── configs/                           # Configuration files
        ├── environments/                  # Environment configs
        └── pipelines/                     # Pipeline configs
```

---

## Learning Path

This project is designed to help you learn these tools:

### Core Tools to Learn

1. **SQL** (Data Engineer persona)
   - PostgreSQL, SQLite
   - Query optimization
   - Database design

2. **PyTorch** (ML Engineer - PyTorch persona)
   - Tensors and autograd
   - Neural network architectures
   - Custom training loops

3. **TensorFlow** (ML Engineer - TensorFlow persona)
   - Keras API
   - tf.data pipelines
   - Model deployment

4. **JAX** (ML Engineer - JAX persona)
   - Functional programming
   - jit, grad, vmap transformations
   - High-performance computing

5. **MLOps Tools** (MLOps Engineer persona)
   - MLflow / Weights & Biases
   - Experiment tracking
   - Reproducibility

6. **Visualization** (Data Viz Specialist persona)
   - matplotlib, seaborn
   - Plotly
   - Streamlit dashboards

---

## Public Data Sources

For portfolio projects, consider these accessible data sources:

### General Purpose
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Government & Open Data
- [Data.gov](https://data.gov/)
- [World Bank Open Data](https://data.worldbank.org/)
- [EU Open Data Portal](https://data.europa.eu/)

### Domain-Specific
- [NOAA Climate Data](https://www.ncdc.noaa.gov/data-access)
- [NIH/NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/)
- [NYC Open Data](https://opendata.cityofnewyork.us/)

---

## How Sessions Work

### Session Start
1. You open workspace and start chatting
2. AI reads personas and context files
3. AI greets you with summary of where you left off
4. AI asks what you want to work on today

### During Session
- AI assumes appropriate persona for task
- Updates session_notes.md with current context
- Every 10 interactions: Checks for patterns to extract
- On major decisions: Updates history/decisions.md

### Save Session
1. You say **"save session"**
2. AI immediately extracts session insights
3. AI updates all relevant files
4. AI clears session_notes.md
5. AI provides session summary

---

## Tips for Best Results

### For Users

1. **Be explicit about what you want to learn**: Tell AI which tools you want to focus on
2. **Invoke personas explicitly**: "As the Data Engineer..." for focused help
3. **Update files regularly**: Keep goals.md and progress.md current
4. **Review user_profile.md**: Check if AI understood you correctly
5. **Save sessions**: Say "save session" when you want to save progress

### For AI Integration

1. **Always read personas file first**: Understand available expert roles
2. **Select appropriate persona**: Based on current task
3. **Update incrementally**: Don't wait until session end
4. **Extract patterns**: Note repeated behaviors
5. **Keep files concise**: Extract insights, not conversations

---

## Getting Started

1. **Clone or use this template**
2. **Open in VSCode** with AI assistant
3. **Start chatting** - AI will guide you through setup
4. **Describe your project idea** - AI helps select appropriate data and approach
5. **Work with personas** - Get expert-level guidance for each aspect

---

**Ready to start?** Open your workspace and begin chatting with the AI!