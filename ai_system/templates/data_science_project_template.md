# Data Science Project Template

**Purpose**: Template for organizing data science projects
**Used by**: Project Architect persona when setting up new data science projects

---

## Suggested Directory Structure

```
project/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, transformed data
│   ├── interim/                # Intermediate data transformations
│   └── external/               # Data from external sources
│
├── notebooks/
│   ├── exploratory/            # EDA and experimentation
│   ├── modeling/               # Model development notebooks
│   └── reports/                # Final analysis notebooks
│
├── src/
│   ├── data/                   # Data loading and processing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation metrics and scripts
│   └── utils/                  # Utility functions
│
├── models/
│   ├── trained/                # Saved model artifacts
│   ├── checkpoints/            # Training checkpoints
│   └── configs/                # Model configuration files
│
├── experiments/
│   ├── logs/                   # Experiment logs
│   └── results/                # Experiment results and metrics
│
├── visualizations/
│   ├── figures/                # Static figures for reports
│   └── dashboards/             # Dashboard configurations
│
├── sql/
│   ├── queries/                # SQL query files
│   ├── schemas/                # Database schema definitions
│   └── migrations/             # Schema migration scripts
│
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── data/                   # Data validation tests
│
├── docs/
│   ├── guides/                 # How-to guides
│   ├── api/                    # API documentation
│   └── research/               # Research notes and papers
│
└── configs/
    ├── environments/           # Environment configurations
    └── pipelines/              # Pipeline configurations
```

---

## Directory Purposes

### data/
Store all data files. Keep raw data immutable - never modify original files.

- **raw/**: Original data exactly as received
- **processed/**: Final, cleaned data ready for modeling
- **interim/**: Intermediate transformations (can be regenerated)
- **external/**: Data from third-party sources

### notebooks/
Jupyter notebooks for exploration and analysis.

- **exploratory/**: Initial data exploration, EDA
- **modeling/**: Model development and experimentation
- **reports/**: Polished notebooks for sharing results

### src/
Production-quality Python code.

- **data/**: Data loading, cleaning, and processing functions
- **features/**: Feature engineering pipelines
- **models/**: Model class definitions
- **training/**: Training loops and scripts
- **evaluation/**: Metrics, evaluation functions
- **utils/**: Shared utility functions

### models/
Model artifacts and configurations.

- **trained/**: Final trained model files
- **checkpoints/**: Training checkpoints for resuming
- **configs/**: Hyperparameters and model configurations

### experiments/
Experiment tracking and results.

- **logs/**: Training logs, TensorBoard logs
- **results/**: Metrics, evaluation results, comparisons

### visualizations/
All visualization outputs.

- **figures/**: Static plots for papers/reports
- **dashboards/**: Streamlit/Dash dashboard code

### sql/
Database-related files.

- **queries/**: Reusable SQL queries
- **schemas/**: Table definitions
- **migrations/**: Schema change scripts

### tests/
Test files for code quality.

- **unit/**: Unit tests for individual functions
- **integration/**: End-to-end pipeline tests
- **data/**: Data validation and quality tests

### docs/
Documentation files.

- **guides/**: Usage guides and tutorials
- **api/**: API reference documentation
- **research/**: Research notes, paper summaries

### configs/
Configuration files.

- **environments/**: Dev, staging, prod configs
- **pipelines/**: Pipeline configuration files

---

## Essential Files

### Root Level Files

```
project/
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment (optional)
├── pyproject.toml             # Modern Python project config
├── .gitignore                 # Git ignore patterns
├── Makefile                   # Common commands
└── README.md                  # Project documentation
```

### .gitignore Template

```gitignore
# Data files (usually too large for git)
data/raw/*
data/processed/*
data/interim/*
!data/*/.gitkeep

# Model files
models/trained/*
models/checkpoints/*
!models/*/.gitkeep

# Experiment outputs
experiments/logs/*
experiments/results/*
!experiments/*/.gitkeep

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.python-version

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## Framework-Specific Additions

### For PyTorch Projects
```
src/
├── models/
│   ├── architectures/         # nn.Module definitions
│   └── losses/                # Custom loss functions
├── training/
│   ├── trainers/              # Training loop classes
│   └── callbacks/             # Training callbacks
```

### For TensorFlow Projects
```
src/
├── models/
│   ├── keras_models/          # Keras model definitions
│   └── custom_layers/         # Custom Keras layers
├── data/
│   └── tf_datasets/           # tf.data pipeline code
```

### For JAX Projects
```
src/
├── models/
│   ├── flax_models/           # Flax nn.Module definitions
│   └── haiku_models/          # Haiku module definitions
├── training/
│   └── train_state/           # TrainState definitions
```

---

## Quick Setup Commands

```bash
# Create directory structure
mkdir -p project/data/{raw,processed,interim,external}
mkdir -p project/notebooks/{exploratory,modeling,reports}
mkdir -p project/src/{data,features,models,training,evaluation,utils}
mkdir -p project/models/{trained,checkpoints,configs}
mkdir -p project/experiments/{logs,results}
mkdir -p project/visualizations/{figures,dashboards}
mkdir -p project/sql/{queries,schemas,migrations}
mkdir -p project/tests/{unit,integration,data}
mkdir -p project/docs/{guides,api,research}
mkdir -p project/configs/{environments,pipelines}

# Create .gitkeep files to preserve empty directories
find project -type d -empty -exec touch {}/.gitkeep \;
```

---

*This template provides a comprehensive structure for data science projects. Adapt based on specific project needs.*