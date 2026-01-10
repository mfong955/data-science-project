"""
Customer Segmentation Analysis
==============================
This script performs K-means clustering to identify distinct customer segments
based on behavioral and demographic features.

As per 03_ANALYSIS_PLAN.md (Analysis 2):
- Identify distinct customer personas
- Understand behavioral differences between segments
- Quantify segment value and characteristics
- Enable targeted strategies

Expected Segments:
- Power Shoppers: High engagement, high conversion, high value
- Window Shoppers: High engagement, low conversion
- Quick Deciders: Low engagement, high conversion
- Deal Seekers: High discount sensitivity, moderate conversion

USAGE:
------
Run from project root:
    python project/notebooks/exploratory/02_customer_segmentation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ============================================================================
# SETUP PATHS
# ============================================================================

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "consumer_behavior_dataset.csv"
FIGURES_PATH = PROJECT_ROOT / "visualizations" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
print(f"Figures will be saved to: {FIGURES_PATH}")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "=" * 60)
print("LOADING AND PREPARING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded {len(df):,} rows and {len(df.columns)} columns")

# ============================================================================
# FEATURE ENGINEERING FOR SEGMENTATION
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create derived features as per 03_ANALYSIS_PLAN.md
# Engagement score = pages_visited * session_duration
df["engagement_score"] = df["pages_visited"] * df["time_spent"]

# Price features
df["price_after_discount"] = df["price"] * (1 - df["discount_applied"] / 100)
df["discount_amount"] = df["price"] * df["discount_applied"] / 100

# Behavioral flags
df["high_engagement"] = (df["pages_visited"] > df["pages_visited"].median()).astype(int)
df["cart_converted"] = (
    (df["add_to_cart"] == 1) & (df["purchase_decision"] == 1)
).astype(int)

# Time per page (engagement efficiency)
df["time_per_page"] = df["time_spent"] / df["pages_visited"].replace(0, 1)

print("Created features:")
print("  - engagement_score (pages_visited * time_spent)")
print("  - price_after_discount")
print("  - discount_amount")
print("  - high_engagement (binary)")
print("  - cart_converted (binary)")
print("  - time_per_page")

# ============================================================================
# SELECT FEATURES FOR CLUSTERING
# ============================================================================

print("\n" + "=" * 60)
print("SELECTING FEATURES FOR CLUSTERING")
print("=" * 60)

# Features for clustering as per 03_ANALYSIS_PLAN.md line 129:
# pages_visited, session_duration (time_spent), price, discount, sentiment_score
clustering_features = [
    "pages_visited",
    "time_spent",
    "price",
    "discount_applied",
    "sentiment_score",
    "rating",
]

print(f"Features selected for clustering: {clustering_features}")

# Create feature matrix
X = df[clustering_features].copy()

# Check for missing values
print(f"\nMissing values per feature:")
print(X.isnull().sum())

# Fill any missing values with median (if any)
X = X.fillna(X.median())

print(f"\nFeature statistics before scaling:")
print(X.describe())

# ============================================================================
# STANDARDIZE FEATURES
# ============================================================================

print("\n" + "=" * 60)
print("STANDARDIZING FEATURES")
print("=" * 60)

# StandardScaler - Standardizes features by removing mean and scaling to unit variance
# z = (x - mean) / std
# This is important for K-means because it uses Euclidean distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)

print("[OK] Features standardized (z-score normalization)")
print(f"\nFeature statistics after scaling:")
print(X_scaled_df.describe())

# ============================================================================
# DETERMINE OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# ============================================================================

print("\n" + "=" * 60)
print("DETERMINING OPTIMAL K (ELBOW METHOD)")
print("=" * 60)

# Test k from 2 to 10
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    # KMeans - K-means clustering algorithm
    # n_clusters: number of clusters to form
    # random_state: for reproducibility
    # n_init: number of times to run with different centroid seeds
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Inertia: sum of squared distances to closest cluster center
    # Lower is better, but decreases with more clusters
    inertias.append(kmeans.inertia_)

    # Silhouette score: measures how similar points are to their own cluster
    # vs other clusters. Range: -1 to 1, higher is better
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)

    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")

# Plot Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot (Inertia)
axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[0].set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
axes[0].set_title("Elbow Method for Optimal k", fontsize=14)
axes[0].grid(True, alpha=0.3)

# Silhouette score plot
axes[1].plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[1].set_ylabel("Silhouette Score", fontsize=12)
axes[1].set_title("Silhouette Score for Different k", fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "segmentation_elbow_method.png", dpi=150, bbox_inches="tight"
)
plt.show()
plt.close()

print(
    f"\n[OK] Elbow method plot saved to: {FIGURES_PATH / 'segmentation_elbow_method.png'}"
)

# Find optimal k based on silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n[OK] Optimal k based on silhouette score: {optimal_k}")

# ============================================================================
# PERFORM K-MEANS CLUSTERING WITH OPTIMAL K
# ============================================================================

print("\n" + "=" * 60)
print(f"PERFORMING K-MEANS CLUSTERING (k={optimal_k})")
print("=" * 60)

# Fit final model with optimal k
# Using k=4 as suggested in analysis plan for interpretable segments
final_k = 4  # Can be changed to optimal_k if preferred
print(f"Using k={final_k} for final clustering (as per analysis plan)")

kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

print(f"\n[OK] Clustering complete")
print(f"\nCluster distribution:")
print(df["cluster"].value_counts().sort_index())

# Calculate final silhouette score
final_silhouette = silhouette_score(X_scaled, df["cluster"])
print(f"\nFinal silhouette score: {final_silhouette:.4f}")

# ============================================================================
# PROFILE EACH CLUSTER
# ============================================================================

print("\n" + "=" * 60)
print("CLUSTER PROFILES")
print("=" * 60)

# Calculate mean values for each cluster
profile_features = clustering_features + [
    "purchase_decision",
    "add_to_cart",
    "abandoned_cart",
    "engagement_score",
    "age",
]

cluster_profiles = df.groupby("cluster")[profile_features].mean()
cluster_sizes = df.groupby("cluster").size()

print("\nCluster Profiles (Mean Values):")
print(cluster_profiles.round(2))

print("\nCluster Sizes:")
print(cluster_sizes)

# ============================================================================
# NAME SEGMENTS BASED ON CHARACTERISTICS
# ============================================================================

print("\n" + "=" * 60)
print("SEGMENT NAMING")
print("=" * 60)

# Analyze each cluster to assign meaningful names
segment_names = {}
segment_descriptions = {}

for cluster_id in range(final_k):
    profile = cluster_profiles.loc[cluster_id]

    # Determine segment characteristics
    high_engagement = (
        profile["engagement_score"] > cluster_profiles["engagement_score"].median()
    )
    high_conversion = (
        profile["purchase_decision"] > cluster_profiles["purchase_decision"].median()
    )
    high_price = profile["price"] > cluster_profiles["price"].median()
    high_discount = (
        profile["discount_applied"] > cluster_profiles["discount_applied"].median()
    )
    high_sentiment = (
        profile["sentiment_score"] > cluster_profiles["sentiment_score"].median()
    )

    # Assign names based on characteristics
    if high_engagement and high_conversion:
        name = "Power Shoppers"
        desc = "High engagement, high conversion, valuable customers"
    elif high_engagement and not high_conversion:
        name = "Window Shoppers"
        desc = "High engagement but low conversion, need nurturing"
    elif not high_engagement and high_conversion:
        name = "Quick Deciders"
        desc = "Low engagement but high conversion, efficient buyers"
    elif high_discount:
        name = "Deal Seekers"
        desc = "Discount-sensitive, respond to promotions"
    else:
        name = f"Segment {cluster_id}"
        desc = "Mixed characteristics"

    segment_names[cluster_id] = name
    segment_descriptions[cluster_id] = desc

    print(f"\nCluster {cluster_id}: {name}")
    print(f"  Description: {desc}")
    print(
        f"  Size: {cluster_sizes[cluster_id]:,} ({cluster_sizes[cluster_id] / len(df) * 100:.1f}%)"
    )
    print(f"  Conversion Rate: {profile['purchase_decision'] * 100:.1f}%")
    print(f"  Avg Engagement: {profile['engagement_score']:.1f}")
    print(f"  Avg Price: ${profile['price']:.2f}")
    print(f"  Avg Discount: {profile['discount_applied']:.1f}%")

# Add segment names to dataframe
df["segment_name"] = df["cluster"].map(segment_names)

# ============================================================================
# VISUALIZE SEGMENTS
# ============================================================================

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Segment Distribution (Pie Chart)
plt.figure(figsize=(10, 8))
colors = plt.cm.Set2(np.linspace(0, 1, final_k))
segment_counts = df["segment_name"].value_counts()
plt.pie(
    segment_counts,
    labels=segment_counts.index,
    autopct="%1.1f%%",
    colors=colors,
    explode=[0.02] * final_k,
    shadow=True,
)
plt.title("Customer Segment Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "segmentation_distribution.png", dpi=150, bbox_inches="tight"
)
plt.show()
plt.close()

# 2. Segment Profiles (Bar Chart)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

metrics_to_plot = [
    ("purchase_decision", "Conversion Rate", True),
    ("engagement_score", "Engagement Score", False),
    ("price", "Average Price ($)", False),
    ("discount_applied", "Discount Applied (%)", False),
    ("sentiment_score", "Sentiment Score", False),
    ("pages_visited", "Pages Visited", False),
]

for idx, (metric, title, is_rate) in enumerate(metrics_to_plot):
    ax = axes[idx]
    values = cluster_profiles[metric]
    if is_rate:
        values = values * 100

    bars = ax.bar(range(final_k), values, color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(final_k))
    ax.set_xticklabels(
        [segment_names[i] for i in range(final_k)], rotation=45, ha="right"
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(values),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig(FIGURES_PATH / "segmentation_profiles.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# 3. Scatter Plot: Engagement vs Conversion by Segment
plt.figure(figsize=(12, 8))
for cluster_id in range(final_k):
    cluster_data = df[df["cluster"] == cluster_id]
    plt.scatter(
        cluster_data["engagement_score"],
        cluster_data["purchase_decision"]
        + np.random.normal(0, 0.02, len(cluster_data)),  # Jitter
        alpha=0.5,
        label=segment_names[cluster_id],
        s=50,
    )

plt.xlabel("Engagement Score (pages × time)", fontsize=12)
plt.ylabel("Purchase Decision (with jitter)", fontsize=12)
plt.title(
    "Customer Segments: Engagement vs Purchase Decision", fontsize=14, fontweight="bold"
)
plt.legend(title="Segment", loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "segmentation_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# 4. Radar Chart for Segment Profiles
def create_radar_chart(profiles, segment_names, features, title):
    """Create a radar chart comparing segment profiles."""
    # Number of features
    num_features = len(features)

    # Compute angle for each feature
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Normalize profiles to 0-1 scale for comparison
    profiles_normalized = profiles[features].copy()
    for col in features:
        min_val = profiles_normalized[col].min()
        max_val = profiles_normalized[col].max()
        if max_val > min_val:
            profiles_normalized[col] = (profiles_normalized[col] - min_val) / (
                max_val - min_val
            )
        else:
            profiles_normalized[col] = 0.5

    colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))

    for idx, (cluster_id, row) in enumerate(profiles_normalized.iterrows()):
        values = row.tolist()
        values += values[:1]  # Complete the loop

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=segment_names[cluster_id],
            color=colors[idx],
        )
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    return fig


radar_features = [
    "pages_visited",
    "time_spent",
    "price",
    "discount_applied",
    "sentiment_score",
    "purchase_decision",
]
fig = create_radar_chart(
    cluster_profiles, segment_names, radar_features, "Segment Comparison Radar Chart"
)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "segmentation_radar.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# 5. Heatmap of Segment Characteristics
plt.figure(figsize=(12, 8))
# Normalize for heatmap
heatmap_data = cluster_profiles[clustering_features + ["purchase_decision"]].copy()
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (
    heatmap_data.max() - heatmap_data.min()
)
heatmap_normalized.index = [segment_names[i] for i in heatmap_normalized.index]

sns.heatmap(
    heatmap_normalized,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Normalized Value"},
)
plt.title(
    "Segment Characteristics Heatmap (Normalized)", fontsize=14, fontweight="bold"
)
plt.xlabel("Features")
plt.ylabel("Segments")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "segmentation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print(f"\n[OK] All visualizations saved to: {FIGURES_PATH}")

# ============================================================================
# STRATEGIC RECOMMENDATIONS PER SEGMENT
# ============================================================================

print("\n" + "=" * 60)
print("STRATEGIC RECOMMENDATIONS")
print("=" * 60)

recommendations = {
    "Power Shoppers": [
        "Implement loyalty program with exclusive benefits",
        "Offer early access to new products",
        "Personalized product recommendations",
        "VIP customer service channel",
    ],
    "Window Shoppers": [
        "Add social proof (reviews, ratings) prominently",
        "Create urgency with limited-time offers",
        "Implement exit-intent popups with discounts",
        "Retargeting campaigns with personalized content",
    ],
    "Quick Deciders": [
        "Streamline checkout process",
        "Offer one-click purchasing",
        "Upsell and cross-sell at checkout",
        "Mobile-optimized experience",
    ],
    "Deal Seekers": [
        "Discount alert notifications",
        "Bundle deals and promotions",
        "Clearance section visibility",
        "Price drop alerts for wishlisted items",
    ],
}

for segment, recs in recommendations.items():
    if segment in segment_names.values():
        print(f"\n{segment}:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")

# ============================================================================
# SAVE SEGMENTED DATA
# ============================================================================

print("\n" + "=" * 60)
print("SAVING SEGMENTED DATA")
print("=" * 60)

# Save the dataframe with cluster assignments
output_path = PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"
df.to_csv(output_path, index=False)
print(f"[OK] Segmented data saved to: {output_path}")

# Save cluster profiles
profiles_path = PROJECT_ROOT / "data" / "processed" / "segment_profiles.csv"
cluster_profiles_named = cluster_profiles.copy()
cluster_profiles_named.index = [segment_names[i] for i in cluster_profiles_named.index]
cluster_profiles_named.to_csv(profiles_path)
print(f"[OK] Segment profiles saved to: {profiles_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 60)
print("SEGMENTATION SUMMARY")
print("=" * 60)

print(f"\nTotal customers analyzed: {len(df):,}")
print(f"Number of segments: {final_k}")
print(f"Silhouette score: {final_silhouette:.4f}")

print("\nSegment Summary:")
summary_df = pd.DataFrame(
    {
        "Segment": [segment_names[i] for i in range(final_k)],
        "Size": [cluster_sizes[i] for i in range(final_k)],
        "Percentage": [
            f"{cluster_sizes[i] / len(df) * 100:.1f}%" for i in range(final_k)
        ],
        "Conversion Rate": [
            f"{cluster_profiles.loc[i, 'purchase_decision'] * 100:.1f}%"
            for i in range(final_k)
        ],
        "Avg Engagement": [
            f"{cluster_profiles.loc[i, 'engagement_score']:.1f}" for i in range(final_k)
        ],
        "Avg Price": [
            f"${cluster_profiles.loc[i, 'price']:.2f}" for i in range(final_k)
        ],
    }
)
print(summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("[OK] Customer segmentation analysis complete!")
print("=" * 60)
