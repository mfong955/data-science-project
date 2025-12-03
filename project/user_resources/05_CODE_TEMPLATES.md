# Code Templates

## 📁 Project Structure Reference

```
project/
├── data/
│   ├── raw/                    # Original data (consumer_behavior_dataset.csv)
│   └── processed/              # Cleaned data, SQLite database
├── notebooks/                  # Jupyter notebooks (01-05)
│   └── exploratory/            # Exploration scripts
├── src/                        # Python modules
│   ├── data/                   # Data loading utilities
│   ├── features/               # Feature engineering
│   ├── models/                 # Model implementations
│   └── visualization/          # Plotting utilities
├── models/                     # Saved model files
├── experiments/                # MLflow experiment tracking
├── visualizations/             # Saved plots and figures
├── sql/queries/                # SQL query files
├── tests/                      # Unit tests
├── configs/                    # Configuration files
└── docs/                       # Documentation
```

---

## 📋 Actual Dataset Columns

```python
COLUMNS = [
    'user_id',           # User identifier
    'product_id',        # Product identifier
    'category',          # Product category
    'price',             # Product price
    'discount_applied',  # Discount percentage
    'payment_method',    # Payment method
    'purchase_date',     # Date of session
    'pages_visited',     # Pages viewed
    'time_spent',        # Session duration
    'add_to_cart',       # Cart addition (0/1)
    'abandoned_cart',    # Cart abandoned (0/1)
    'rating',            # Product rating
    'review_text',       # Review content
    'sentiment_score',   # Sentiment score
    'age',               # Customer age
    'gender',            # Customer gender
    'income_level',      # Income category
    'location',          # Customer location
    'purchase_decision', # TARGET (0/1)
]
```

---

## 🐍 PYTHON TEMPLATES

### Template 1: Standard Notebook Header
```python
"""
Notebook: [01_data_exploration.ipynb]
Author: Matthew Fong
Date: 2024-XX-XX
Purpose: [Brief description]

This notebook covers:
1. [Section 1]
2. [Section 2]
3. [Section 3]
"""

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths - using pathlib for cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjust based on file location
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
VIZ_DIR = PROJECT_ROOT / 'visualizations'

# Verify paths exist
print(f"Raw data path exists: {DATA_RAW.exists()}")
print(f"Processed data path exists: {DATA_PROCESSED.exists()}")

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

---

### Template 2: Data Loading
```python
def load_data(filepath=None):
    """
    Load the consumer behavior dataset.
    
    Parameters:
    -----------
    filepath : str or Path, optional
        Path to CSV file. Defaults to project raw data location.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if filepath is None:
        filepath = DATA_RAW / 'consumer_behavior_dataset.csv'
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# Usage
df = load_data()
df.head()
```

---

### Template 3: Quick EDA Function
```python
def quick_eda(df, target_col='purchase_decision'):
    """
    Perform quick exploratory data analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    target_col : str
        Name of target variable
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print(f"\n📋 Data Types:")
    print(df.dtypes.value_counts())
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    print(f"\n🎯 Target Distribution ({target_col}):")
    if target_col in df.columns:
        print(df[target_col].value_counts(normalize=True).round(3))
    
    print(f"\n📈 Numeric Summary:")
    print(df.describe().round(2))
    
    print("\n" + "=" * 60)

# Usage
quick_eda(df)
```

---

### Template 4: Feature Engineering Pipeline
```python
def engineer_features(df):
    """
    Create engineered features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    
    Returns:
    --------
    pd.DataFrame
        Dataset with new features
    """
    df = df.copy()
    
    # Parse date features
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['day_of_week'] = df['purchase_date'].dt.day_name()
    df['month'] = df['purchase_date'].dt.month
    df['is_weekend'] = (df['purchase_date'].dt.dayofweek >= 5).astype(int)
    
    # Engagement features
    df['engagement_score'] = df['pages_visited'] * df['time_spent']
    df['pages_per_minute'] = df['pages_visited'] / df['time_spent'].clip(lower=1)
    
    # Price features
    df['price_after_discount'] = df['price'] * (1 - df['discount_applied'] / 100)
    df['discount_amount'] = df['price'] * df['discount_applied'] / 100
    
    # Behavioral flags
    df['high_engagement'] = (df['engagement_score'] > df['engagement_score'].median()).astype(int)
    df['cart_converted'] = ((df['add_to_cart'] == 1) & (df['purchase_decision'] == 1)).astype(int)
    
    # Sentiment categories
    df['sentiment_category'] = pd.cut(
        df['sentiment_score'], 
        bins=[-float('inf'), 0, 0.5, float('inf')],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    print(f"✓ Created {len(df.columns)} total columns")
    
    return df

# Usage
df_features = engineer_features(df)
```

---

### Template 5: Train/Test Split with Stratification
```python
from sklearn.model_selection import train_test_split

def prepare_data_splits(df, target_col='purchase_decision', test_size=0.2, val_size=0.1):
    """
    Prepare train/validation/test splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target_col : str
        Name of target column
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from training data)
    
    Returns:
    --------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Define feature columns (exclude non-predictive columns)
    exclude_cols = ['user_id', 'product_id', 'purchase_date', 'review_text', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=RANDOM_SEED
    )
    
    # Second split: train vs val
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_adjusted, 
        stratify=y_temp, 
        random_state=RANDOM_SEED
    )
    
    print(f"✓ Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"✓ Val:   {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"✓ Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(df_features)
```

---

### Template 6: Preprocessing Pipeline
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessor(X_train):
    """
    Create preprocessing pipeline for mixed data types.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    
    Returns:
    --------
    ColumnTransformer
        Fitted preprocessor
    """
    # Identify column types
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', 'passthrough', categorical_cols)  # Or use OneHotEncoder
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_cols, categorical_cols

# Usage
preprocessor, numeric_cols, categorical_cols = create_preprocessor(X_train)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
```

---

## 🔥 PYTORCH TEMPLATES

### Template 7: PyTorch Dataset Class
```python
import torch
from torch.utils.data import Dataset, DataLoader

class EcommerceDataset(Dataset):
    """PyTorch Dataset for e-commerce data."""
    
    def __init__(self, X, y):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features
        y : np.ndarray or pd.Series
            Labels
        """
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Usage
train_dataset = EcommerceDataset(X_train_processed, y_train.values)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Verify
sample_X, sample_y = next(iter(train_loader))
print(f"Batch X shape: {sample_X.shape}")
print(f"Batch y shape: {sample_y.shape}")
```

---

### Template 8: PyTorch Neural Network
```python
import torch.nn as nn
import torch.nn.functional as F

class PurchasePredictor(nn.Module):
    """Neural network for purchase prediction."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        """
        Initialize network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list
            Sizes of hidden layers
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Usage
input_dim = X_train_processed.shape[1]
model = PurchasePredictor(input_dim, hidden_dims=[64, 32, 16])
print(model)
```

---

### Template 9: PyTorch Training Loop
```python
from sklearn.metrics import roc_auc_score

def train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Train PyTorch model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    
    Returns:
    --------
    dict
        Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                val_loss += criterion(y_pred, y_batch).item()
                val_preds.extend(y_pred.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_true, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val AUC: {val_auc:.4f}")
    
    return history

# Usage
history = train_pytorch_model(model, train_loader, val_loader, epochs=50)
```

---

## 🧠 TENSORFLOW TEMPLATES

### Template 10: TensorFlow/Keras Model
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_tf_model(input_dim, hidden_dims=[64, 32], dropout=0.3):
    """
    Create TensorFlow/Keras model.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_dims : list
        Sizes of hidden layers
    dropout : float
        Dropout rate
    
    Returns:
    --------
    keras.Model
        Compiled model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
    ])
    
    for hidden_dim in hidden_dims:
        model.add(layers.Dense(hidden_dim, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

# Usage
tf_model = create_tf_model(input_dim, hidden_dims=[64, 32, 16])
tf_model.summary()
```

---

### Template 11: TensorFlow Training with Callbacks
```python
def train_tf_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train TensorFlow model with callbacks.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / 'tf_best_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Usage
history = train_tf_model(tf_model, X_train_processed, y_train, X_val_processed, y_val)
```

---

## ⚡ JAX TEMPLATES

### Template 12: JAX Neural Network with Flax
```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class JAXPurchasePredictor(nn.Module):
    """JAX/Flax neural network for purchase prediction."""
    
    hidden_dims: tuple = (64, 32)
    dropout_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)

# Initialize model
def create_jax_model(input_dim, hidden_dims=(64, 32)):
    """Create and initialize JAX model."""
    model = JAXPurchasePredictor(hidden_dims=hidden_dims)
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(RANDOM_SEED)
    dummy_input = jnp.ones((1, input_dim))
    variables = model.init(key, dummy_input)
    
    return model, variables

# Usage
jax_model, variables = create_jax_model(input_dim)
```

---

## 📊 VISUALIZATION TEMPLATES

### Template 13: Model Comparison Plot
```python
def plot_model_comparison(results_dict, metric='auc'):
    """
    Plot comparison of multiple models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results_dict.keys())
    x = np.arange(len(models))
    width = 0.25
    
    train_scores = [results_dict[m]['train'] for m in models]
    val_scores = [results_dict[m]['val'] for m in models]
    test_scores = [results_dict[m]['test'] for m in models]
    
    ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
    ax.bar(x, val_scores, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_scores, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison - {metric.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'model_comparison_{metric}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

### Template 14: Feature Importance Plot
```python
def plot_feature_importance(importance_df, top_n=15, title='Feature Importance'):
    """
    Plot feature importance.
    """
    df_plot = importance_df.nlargest(top_n, 'importance')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_plot)))
    
    ax.barh(df_plot['feature'], df_plot['importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage (with sklearn model)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_model.fit(X_train_processed, y_train)

importance_df = pd.DataFrame({
    'feature': numeric_cols + categorical_cols,  # Use actual feature names
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plot_feature_importance(importance_df)
```

---

## 🧪 EVALUATION TEMPLATES

### Template 15: Comprehensive Model Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

def evaluate_model(y_true, y_pred_proba, threshold=0.5, model_name='Model'):
    """
    Comprehensive model evaluation.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba)
    }
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        print(f"{metric.upper():12s}: {value:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics

# Usage
metrics = evaluate_model(y_test, y_pred_proba, model_name='PyTorch NN')
```

---

## 💾 MODEL SAVING TEMPLATES

### Template 16: Save/Load Models
```python
import joblib
import json

# Scikit-learn models
def save_sklearn_model(model, name, metrics=None):
    """Save sklearn model with metadata."""
    model_path = MODELS_DIR / f'{name}.joblib'
    joblib.dump(model, model_path)
    
    if metrics:
        meta_path = MODELS_DIR / f'{name}_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"✓ Model saved to {model_path}")

# PyTorch models
def save_pytorch_model(model, name, metrics=None):
    """Save PyTorch model."""
    model_path = MODELS_DIR / f'{name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, model_path)
    print(f"✓ Model saved to {model_path}")

# TensorFlow models
def save_tf_model(model, name):
    """Save TensorFlow model."""
    model_path = MODELS_DIR / f'{name}.keras'
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
```

---

## 📝 Quick Reference

### Import Blocks

**Data Science Basics:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
```

**Machine Learning:**
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
```

**Deep Learning - PyTorch:**
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

**Deep Learning - TensorFlow:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

**Deep Learning - JAX:**
```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
```

---

## 📁 Where to Save Code

- **Notebooks:** `project/notebooks/`
- **Reusable modules:** `project/src/`
- **Saved models:** `project/models/`
- **Visualizations:** `project/visualizations/`

---

**Next Step:** Read 06_RESUME_BULLETS.md for portfolio presentation!