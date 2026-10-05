"""
Customer segmentation with a Kohonen Self-Organizing Map (SOM) + K-means.

Pipeline: synthetic retail data -> z-score scaling -> 10x10 SOM -> K-means (k=4)
on the SOM neurons -> segment profiles -> quality metrics -> figures.

Usage:
    python som_segmentation.py
Outputs go to outputs/ (figures, CSVs).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT_DIR = "outputs"
DATA_PATH = "data/customers.csv"
FEATURES = [
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Transaction_Value",
    "Customer_Tenure",
    "Product_Diversity",
]
GRID = (10, 10)
SIGMA, LEARNING_RATE, ITERATIONS = 1.0, 0.5, 10_000
N_CLUSTERS = 4


# ---------------------------------------------------------------- data
def create_customer_dataset(n_per_segment=125):
    """Synthetic retail customers drawn from four behavioural archetypes.

    Each tuple is (mean, std) for: income (k$), spending score, visits/month,
    avg transaction ($), tenure (months), product diversity.
    """
    archetypes = {
        1: [(80, 15), (75, 10), (12, 2), (200, 30), (36, 8), (8, 1)],      # high-value loyal
        2: [(35, 10), (30, 8), (15, 3), (50, 15), (24, 10), (4, 1)],       # budget-conscious
        3: [(90, 20), (60, 12), (4, 1), (400, 60), (18, 6), (6, 1.5)],     # occasional high spenders
        4: [(55, 15), (45, 10), (8, 2), (120, 25), (6, 3), (5, 1.5)],      # new customers
    }
    np.random.seed(SEED)  # same draw order as the original analysis, so results are reproducible
    frames = []
    for seg, params in archetypes.items():
        cols = {name: np.random.normal(mu, sd, n_per_segment) for name, (mu, sd) in zip(FEATURES, params)}
        cols["True_Segment"] = seg
        frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    df[FEATURES] = df[FEATURES].abs()  # values must be positive
    return df


# ---------------------------------------------------------------- model
def train_som(data):
    som = MiniSom(GRID[0], GRID[1], data.shape[1], sigma=SIGMA, learning_rate=LEARNING_RATE,
                  neighborhood_function="gaussian", random_seed=SEED)
    som.random_weights_init(data)
    som.train_random(data, ITERATIONS)
    return som


def cluster_neurons(som, data, n_clusters=N_CLUSTERS):
    """K-means on the neuron weight vectors, then map each sample to its BMU's cluster."""
    weights = som.get_weights()
    flat = weights.reshape(-1, weights.shape[2])
    labels = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(flat)
    cluster_map = labels.reshape(weights.shape[0], weights.shape[1])
    sample_clusters = np.array([cluster_map[som.winner(x)] for x in data])
    return sample_clusters, cluster_map


def evaluate(som, data, sample_clusters):
    weights = som.get_weights()
    flat = weights.reshape(-1, weights.shape[2])
    dists = np.linalg.norm(data[:, None, :] - flat[None, :, :], axis=2)   # (n_samples, n_neurons)

    order = np.argsort(dists, axis=1)
    quantization_error = dists[np.arange(len(data)), order[:, 0]].mean()

    ncols = weights.shape[1]
    bmu = np.stack(divmod(order[:, 0], ncols), axis=1)
    second = np.stack(divmod(order[:, 1], ncols), axis=1)
    not_adjacent = (np.abs(bmu - second) > 1).any(axis=1)
    topographic_error = not_adjacent.mean()

    return {
        "Quantization Error": quantization_error,
        "Topographic Error": topographic_error,
        "Silhouette Score": silhouette_score(data, sample_clusters),
        "Davies-Bouldin Index": davies_bouldin_score(data, sample_clusters),
    }


def name_segment(profile):
    """Rule-based business label for a segment's mean feature values."""
    if profile["Annual_Income"] > 70 and profile["Spending_Score"] > 65 and profile["Customer_Tenure"] > 30:
        return "High-Value Loyal Customers"
    if profile["Annual_Income"] < 45 and profile["Avg_Transaction_Value"] < 100:
        return "Budget-Conscious Shoppers"
    if profile["Avg_Transaction_Value"] > 300 and profile["Purchase_Frequency"] < 6:
        return "Occasional High Spenders"
    return "Emerging / New Customers"


# ---------------------------------------------------------------- plots
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all(df, som, data, cluster_map, sample_clusters):
    # feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, f in zip(axes.ravel(), FEATURES):
        ax.hist(df[f], bins=30, color="steelblue", alpha=0.75, edgecolor="black")
        ax.set_title(f.replace("_", " "))
    _save(fig, "01_feature_distributions.png")

    # correlation matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation matrix")
    _save(fig, "02_correlation_matrix.png")

    # U-matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(som.distance_map().T, cmap="bone_r", origin="lower")
    fig.colorbar(im, ax=ax, label="Average distance to neighbours")
    ax.set_title("U-Matrix")
    _save(fig, "03_umatrix.png")

    # component planes
    weights = som.get_weights()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, (ax, f) in enumerate(zip(axes.ravel(), FEATURES)):
        im = ax.imshow(weights[:, :, i].T, cmap="viridis", origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_title(f.replace("_", " "))
    _save(fig, "04_component_planes.png")

    # hit map
    hits = np.zeros(GRID)
    for x in data:
        hits[som.winner(x)] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(hits.T, cmap="YlOrRd", origin="lower")
    fig.colorbar(im, ax=ax, label="Samples per neuron")
    ax.set_title("Hit map")
    _save(fig, "05_hit_map.png")

    # cluster map
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cluster_map.T, cmap="Set3", origin="lower")
    ax.set_title("SOM cluster map (K-means on neurons)")
    _save(fig, "06_cluster_map.png")

    # segments in feature space
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    pairs = [("Annual_Income", "Spending_Score"), ("Purchase_Frequency", "Avg_Transaction_Value"),
             ("Customer_Tenure", "Product_Diversity")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for c in np.unique(sample_clusters):
            m = sample_clusters == c
            ax.scatter(df.loc[m, a], df.loc[m, b], c=colors[int(c)], s=25, alpha=0.6, label=f"Segment {int(c) + 1}")
        ax.set_xlabel(a.replace("_", " "))
        ax.set_ylabel(b.replace("_", " "))
        ax.legend()
    counts = pd.Series(sample_clusters).value_counts().sort_index()
    axes[1, 1].bar([f"Segment {i + 1}" for i in counts.index], counts.values, color=colors[: len(counts)])
    axes[1, 1].set_ylabel("Customers")
    _save(fig, "07_customer_segments.png")


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    df = create_customer_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset: {df.shape[0]} customers, {len(FEATURES)} features -> {DATA_PATH}")

    data = StandardScaler().fit_transform(df[FEATURES].values)
    som = train_som(data)
    sample_clusters, cluster_map = cluster_neurons(som, data)

    metrics = evaluate(som, data, sample_clusters)
    print("\nModel quality")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "som_metrics.csv"), index=False)

    result = df.copy()
    result["Segment"] = sample_clusters + 1
    profiles = result.groupby("Segment")[FEATURES].mean()
    profiles["Customers"] = result.groupby("Segment").size()
    profiles["Share_%"] = (profiles["Customers"] / len(result) * 100).round(1)
    profiles["Label"] = [name_segment(row) for _, row in profiles.iterrows()]
    print("\nSegment profiles")
    print(profiles.round(2).to_string())
    profiles.round(2).to_csv(os.path.join(OUT_DIR, "segment_profiles.csv"))
    result.to_csv(os.path.join(OUT_DIR, "customer_segments.csv"), index=False)

    plot_all(df, som, data, cluster_map, sample_clusters)
    print(f"\nFigures and CSVs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
