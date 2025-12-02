# Python Code Templates

## 📦 Standard Imports

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.cluster import KMeans

# Statistics
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

# Database
import sqlite3
from sqlalchemy import create_engine

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

---

## 📊 Template 1: Data Loading & Initial Exploration

```python
# Load data
def load_data(filepath='data/consumer_behavior_dataset.csv'):
    """Load and perform initial data checks"""
    df = pd.read_csv(filepath)
    
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(df.dtypes)
    
    print("\n" + "="*50)
    print("MISSING VALUES")
    print("="*50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    
    print("\n" + "="*50)
    print("SAMPLE DATA")
    print("="*50)
    print(df.head())
    
    return df

# Usage
df = load_data()
```

---

## 📈 Template 2: Calculate Key Metrics

```python
def calculate_key_metrics(df):
    """Calculate primary product analytics metrics"""
    
    metrics = {
        # Conversion metrics
        'Total Sessions': len(df),
        'Total Purchases': df['purchase_decision'].sum(),
        'Conversion Rate (%)': (df['purchase_decision'].mean() * 100),
        
        # Revenue metrics
        'Total Revenue': df[df['purchase_decision']==1]['price'].sum(),
        'Average Order Value': df[df['purchase_decision']==1]['price'].mean(),
        'Revenue per Session': df[df['purchase_decision']==1]['price'].sum() / len(df),
        
        # Engagement metrics
        'Avg Pages Visited': df['pages_visited'].mean(),
        'Avg Session Duration (min)': df['session_duration'].mean(),
        'Bounce Rate (%)': (df['pages_visited'] == 1).mean() * 100,
        
        # Cart metrics
        'Cart Addition Rate (%)': (df['cart_added'].mean() * 100),
        'Cart Abandonment Rate (%)': (
            df[df['cart_added']==1]['purchase_decision'].apply(lambda x: 1 if x==0 else 0).mean() * 100
        ),
    }
    
    # Create formatted DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df['Value'] = metrics_df['Value'].round(2)
    
    print("="*50)
    print("KEY METRICS")
    print("="*50)
    print(metrics_df)
    
    return metrics_df

# Usage
metrics = calculate_key_metrics(df)
```

---

## 📊 Template 3: Visualizations

```python
def create_conversion_funnel(df):
    """Create conversion funnel visualization"""
    
    funnel_data = {
        'Stage': ['All Sessions', 'Browsed (2+ pages)', 'Added to Cart', 'Purchased'],
        'Count': [
            len(df),
            len(df[df['pages_visited'] >= 2]),
            df['cart_added'].sum(),
            df['purchase_decision'].sum()
        ]
    }
    
    fig = px.funnel(
        funnel_data, 
        x='Count', 
        y='Stage',
        title='E-Commerce Conversion Funnel',
        color='Stage'
    )
    
    fig.update_layout(height=500)
    fig.show()
    
    return funnel_data

def plot_conversion_by_category(df, category_col='product_category'):
    """Plot conversion rate by category"""
    
    conv_by_cat = df.groupby(category_col).agg({
        'purchase_decision': ['count', 'sum', 'mean']
    }).round(3)
    
    conv_by_cat.columns = ['Sessions', 'Purchases', 'Conversion_Rate']
    conv_by_cat = conv_by_cat.reset_index()
    conv_by_cat['Conversion_Rate'] = conv_by_cat['Conversion_Rate'] * 100
    
    fig = px.bar(
        conv_by_cat.sort_values('Conversion_Rate', ascending=False),
        x=category_col,
        y='Conversion_Rate',
        title=f'Conversion Rate by {category_col}',
        text='Conversion_Rate',
        color='Conversion_Rate',
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500)
    fig.show()
    
    return conv_by_cat

def plot_distribution_comparison(df, feature, title=None):
    """Compare feature distribution between converters and non-converters"""
    
    fig = px.box(
        df,
        x='purchase_decision',
        y=feature,
        color='purchase_decision',
        title=title or f'{feature} Distribution: Purchasers vs Non-Purchasers',
        labels={'purchase_decision': 'Purchase Decision', feature: feature}
    )
    
    fig.update_layout(showlegend=False, height=400)
    fig.show()

# Usage
create_conversion_funnel(df)
plot_conversion_by_category(df, 'product_category')
plot_distribution_comparison(df, 'session_duration', 'Session Duration by Purchase Decision')
```

---

## 🎯 Template 4: Customer Segmentation

```python
def perform_customer_segmentation(df, n_clusters=4):
    """Perform K-means clustering for customer segmentation"""
    
    # Select features for clustering
    feature_cols = ['pages_visited', 'session_duration', 'price', 'discount', 'sentiment_score']
    
    # Handle missing values
    X = df[feature_cols].fillna(df[feature_cols].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    # Analyze segments
    segment_summary = df.groupby('segment').agg({
        'purchase_decision': ['count', 'mean'],
        'price': 'mean',
        'pages_visited': 'mean',
        'session_duration': 'mean',
        'discount': 'mean',
        'sentiment_score': 'mean'
    }).round(2)
    
    segment_summary.columns = [
        'Count', 'Conversion_Rate', 'Avg_Price', 
        'Avg_Pages', 'Avg_Duration', 'Avg_Discount', 'Avg_Sentiment'
    ]
    
    segment_summary['Conversion_Rate'] = segment_summary['Conversion_Rate'] * 100
    
    print("="*50)
    print("CUSTOMER SEGMENTS")
    print("="*50)
    print(segment_summary)
    
    # Assign meaningful names based on characteristics
    segment_names = {}
    for seg in range(n_clusters):
        seg_data = segment_summary.loc[seg]
        if seg_data['Avg_Pages'] > 10 and seg_data['Conversion_Rate'] > 20:
            segment_names[seg] = 'Power Shoppers'
        elif seg_data['Avg_Pages'] > 10 and seg_data['Conversion_Rate'] < 15:
            segment_names[seg] = 'Window Shoppers'
        elif seg_data['Avg_Pages'] < 5 and seg_data['Conversion_Rate'] > 15:
            segment_names[seg] = 'Quick Deciders'
        else:
            segment_names[seg] = 'Deal Seekers'
    
    df['segment_name'] = df['segment'].map(segment_names)
    
    print("\n" + "="*50)
    print("SEGMENT NAMES")
    print("="*50)
    for seg, name in segment_names.items():
        print(f"Segment {seg}: {name}")
    
    return df, segment_summary, segment_names

# Usage
df, segment_summary, segment_names = perform_customer_segmentation(df, n_clusters=4)
```

---

## 🤖 Template 5: Predictive Modeling

```python
def build_purchase_prediction_model(df):
    """Build and evaluate models to predict purchase decisions"""
    
    # Feature engineering
    df_model = df.copy()
    df_model['engagement_score'] = df_model['pages_visited'] * df_model['session_duration']
    df_model['time_per_page'] = df_model['session_duration'] / df_model['pages_visited'].replace(0, 1)
    df_model['has_discount'] = (df_model['discount'] > 0).astype(int)
    
    # Select features
    feature_cols = [
        'age', 'pages_visited', 'session_duration', 'price', 'discount',
        'sentiment_score', 'cart_added', 'engagement_score', 'time_per_page', 'has_discount'
    ]
    
    # Prepare data
    X = df_model[feature_cols].fillna(df_model[feature_cols].mean())
    y = df_model['purchase_decision']
    
    # Add categorical features
    categorical_cols = ['gender', 'product_category', 'location', 'income_level']
    for col in categorical_cols:
        if col in df_model.columns:
            dummies = pd.get_dummies(df_model[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n{name}:")
        print(f"  ROC-AUC: {results[name]['ROC-AUC']:.3f}")
        print(f"  Accuracy: {results[name]['Accuracy']:.3f}")
        print(f"  Precision: {results[name]['Precision']:.3f}")
        print(f"  Recall: {results[name]['Recall']:.3f}")
        print(f"  F1 Score: {results[name]['F1']:.3f}")
    
    # Feature importance (from Random Forest)
    best_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print("\n" + "="*50)
    print("TOP 10 FEATURE IMPORTANCE (Random Forest)")
    print("="*50)
    print(feature_importance)
    
    # Plot feature importance
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    fig.show()
    
    return models, results, X_train, X_test, y_train, y_test

# Usage
models, results, X_train, X_test, y_train, y_test = build_purchase_prediction_model(df)
```

---

## 📊 Template 6: A/B Test Analysis

```python
def analyze_ab_test(df, treatment_col, treatment_values, metric='purchase_decision'):
    """Perform statistical A/B test analysis"""
    
    # Split into control and treatment
    control_data = df[df[treatment_col] == treatment_values[0]][metric]
    treatment_data = df[df[treatment_col] == treatment_values[1]][metric]
    
    # Calculate statistics
    n_control = len(control_data)
    n_treatment = len(treatment_data)
    
    mean_control = control_data.mean()
    mean_treatment = treatment_data.mean()
    
    std_control = control_data.std()
    std_treatment = treatment_data.std()
    
    # Two-proportion z-test (for binary outcomes)
    if metric == 'purchase_decision':
        # Calculate pooled proportion
        p_pooled = (control_data.sum() + treatment_data.sum()) / (n_control + n_treatment)
        
        # Calculate standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        
        # Calculate z-statistic
        z_stat = (mean_treatment - mean_control) / se
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Calculate confidence interval
        ci_95 = 1.96 * se
        ci_lower = (mean_treatment - mean_control) - ci_95
        ci_upper = (mean_treatment - mean_control) + ci_95
        
    else:
        # T-test for continuous outcomes
        t_stat, p_value = ttest_ind(treatment_data, control_data)
        se = np.sqrt(std_control**2/n_control + std_treatment**2/n_treatment)
        ci_95 = 1.96 * se
        ci_lower = (mean_treatment - mean_control) - ci_95
        ci_upper = (mean_treatment - mean_control) + ci_95
    
    # Calculate effect size
    absolute_lift = mean_treatment - mean_control
    relative_lift = (absolute_lift / mean_control) * 100 if mean_control != 0 else 0
    
    # Print results
    print("="*60)
    print(f"A/B TEST ANALYSIS: {treatment_col}")
    print("="*60)
    print(f"\nControl Group ({treatment_values[0]}):")
    print(f"  Sample Size: {n_control:,}")
    print(f"  Mean: {mean_control:.4f} ({mean_control*100:.2f}%)")
    
    print(f"\nTreatment Group ({treatment_values[1]}):")
    print(f"  Sample Size: {n_treatment:,}")
    print(f"  Mean: {mean_treatment:.4f} ({mean_treatment*100:.2f}%)")
    
    print(f"\nTest Results:")
    print(f"  Absolute Difference: {absolute_lift:.4f} ({absolute_lift*100:.2f} percentage points)")
    print(f"  Relative Lift: {relative_lift:.2f}%")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Statistically Significant (α=0.05): {'YES ✓' if p_value < 0.05 else 'NO ✗'}")
    
    # Visualization
    comparison_data = pd.DataFrame({
        'Group': [treatment_values[0], treatment_values[1]],
        'Conversion Rate': [mean_control * 100, mean_treatment * 100],
        'Sample Size': [n_control, n_treatment]
    })
    
    fig = px.bar(
        comparison_data,
        x='Group',
        y='Conversion Rate',
        text='Conversion Rate',
        title=f'A/B Test: {treatment_col}',
        color='Group'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    fig.show()
    
    return {
        'control_mean': mean_control,
        'treatment_mean': mean_treatment,
        'absolute_lift': absolute_lift,
        'relative_lift': relative_lift,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05
    }

# Usage Example 1: Discount effect
df['discount_group'] = df['discount'].apply(lambda x: 'High Discount' if x >= 15 else 'Low Discount')
discount_test = analyze_ab_test(df, 'discount_group', ['Low Discount', 'High Discount'])

# Usage Example 2: Engagement effect
df['engagement_group'] = df['pages_visited'].apply(lambda x: 'High Engagement' if x >= 10 else 'Low Engagement')
engagement_test = analyze_ab_test(df, 'engagement_group', ['Low Engagement', 'High Engagement'])
```

---

## 📈 Template 7: Business Impact Calculator

```python
def calculate_business_impact(df, current_conversion_rate, lift_pct, avg_order_value=None):
    """Calculate business impact of improvements"""
    
    if avg_order_value is None:
        avg_order_value = df[df['purchase_decision']==1]['price'].mean()
    
    total_sessions = len(df)
    current_purchases = int(total_sessions * current_conversion_rate)
    current_revenue = current_purchases * avg_order_value
    
    new_conversion_rate = current_conversion_rate * (1 + lift_pct/100)
    new_purchases = int(total_sessions * new_conversion_rate)
    new_revenue = new_purchases * avg_order_value
    
    incremental_purchases = new_purchases - current_purchases
    incremental_revenue = new_revenue - current_revenue
    
    print("="*60)
    print("BUSINESS IMPACT CALCULATOR")
    print("="*60)
    print(f"\nCurrent State:")
    print(f"  Total Sessions: {total_sessions:,}")
    print(f"  Conversion Rate: {current_conversion_rate*100:.2f}%")
    print(f"  Purchases: {current_purchases:,}")
    print(f"  Revenue: ${current_revenue:,.2f}")
    
    print(f"\nProjected State (with {lift_pct:.1f}% lift):")
    print(f"  Conversion Rate: {new_conversion_rate*100:.2f}%")
    print(f"  Purchases: {new_purchases:,}")
    print(f"  Revenue: ${new_revenue:,.2f}")
    
    print(f"\nIncremental Impact:")
    print(f"  Additional Purchases: {incremental_purchases:,} (+{(incremental_purchases/current_purchases)*100:.1f}%)")
    print(f"  Additional Revenue: ${incremental_revenue:,.2f} (+{lift_pct:.1f}%)")
    
    # Annual projection (assuming dataset represents typical period)
    print(f"\nAnnualized Projection (if sustained):")
    print(f"  Annual Additional Revenue: ${incremental_revenue * 12:,.2f}")
    
    return {
        'incremental_purchases': incremental_purchases,
        'incremental_revenue': incremental_revenue,
        'annual_revenue_opportunity': incremental_revenue * 12
    }

# Usage
current_cr = df['purchase_decision'].mean()
impact = calculate_business_impact(df, current_cr, lift_pct=15)
```

---

## 💾 Template 8: Save Results

```python
def save_analysis_results(df, metrics, segment_summary, model_results, output_dir='results'):
    """Save all analysis results"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed dataset
    df.to_csv(f'{output_dir}/processed_data.csv', index=False)
    print(f"✓ Saved: {output_dir}/processed_data.csv")
    
    # Save metrics
    metrics.to_csv(f'{output_dir}/key_metrics.csv')
    print(f"✓ Saved: {output_dir}/key_metrics.csv")
    
    # Save segments
    segment_summary.to_csv(f'{output_dir}/segment_summary.csv')
    print(f"✓ Saved: {output_dir}/segment_summary.csv")
    
    # Save model results
    model_results_df = pd.DataFrame(model_results).T
    model_results_df.to_csv(f'{output_dir}/model_results.csv')
    print(f"✓ Saved: {output_dir}/model_results.csv")
    
    print(f"\n✓ All results saved to '{output_dir}/' directory")

# Usage
save_analysis_results(df, metrics, segment_summary, results)
```

---

## ✅ Quick Start Checklist

Use this checklist when starting a new notebook:

```python
# Checklist for each notebook
"""
[ ] Import all required libraries
[ ] Load data with quality checks
[ ] Calculate baseline metrics
[ ] Create exploratory visualizations
[ ] Perform primary analysis
[ ] Generate insights and recommendations
[ ] Save results
[ ] Document findings in markdown cells
"""
```

---

**Next Step:** Read 06_RESUME_BULLETS.md for career materials!