"""
K-Means Clustering Concept & Visualization
Unsupervised clustering featuring Cluster Partitions, Centroid Tracking, and Silhouette/Elbow validation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


def run_kmeans(show_plot: bool = True, save_dir: str = "plots", n_clusters: int = 3):
    print("=" * 60)
    print(f" [K-MEANS CLUSTERING] Unsupervised Customer Segmentation (k = {n_clusters})")
    print("=" * 60)

    # 1. Dataset Generation: 3 realistic customer behavioral clusters (Spending Score vs Visit Frequency)
    np.random.seed(42)
    c1 = np.random.normal(loc=[25, 30], scale=[6, 7], size=(45, 2))  # Low spend, low visits: "Occasional"
    c2 = np.random.normal(loc=[70, 45], scale=[7, 8], size=(50, 2))  # High spend, moderate visits: "High Value"
    c3 = np.random.normal(loc=[45, 80], scale=[6, 6], size=(40, 2))  # Moderate spend, high visits: "Frequent Shoppers"

    X = np.vstack((c1, c2, c3))

    # 2. Elbow & Silhouette Analysis across range of k (k = 2 to 6)
    k_range = list(range(2, 7))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        score = silhouette_score(X, km.labels_)
        silhouettes.append(score)

    # 3. Model Training with chosen k
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    current_silhouette = silhouette_score(X, labels)

    print(f"Dataset Size: {len(X)} samples across 2 features")
    print(f"K-Means Inertia (WCSS): {model.inertia_:.1f}")
    print(f"Silhouette Score (k={n_clusters}): {current_silhouette:.3f}")
    print("Cluster Centroid Coordinates [Spending, Frequency]:")
    cluster_names = ["Cluster 0 (Occasional)", "Cluster 1 (High-Value)", "Cluster 2 (Frequent)"]
    for i, c in enumerate(centroids):
        name = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
        sample_count = np.sum(labels == i)
        print(f"  {name}: Centroid = ({c[0]:.1f}, {c[1]:.1f}) | Members = {sample_count}")
    print()

    # 4. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax_clusters, ax_metrics) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Subplot 1: Voronoi Decision Surface & Clusters
    x_min, x_max = X[:, 0].min() - 8, X[:, 0].max() + 8
    y_min, y_max = X[:, 1].min() - 8, X[:, 1].max() + 8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Cluster background
    ax_clusters.contourf(xx, yy, Z, alpha=0.18, cmap="Set1")
    ax_clusters.contour(xx, yy, Z, colors="#4B5563", linewidths=1.2, linestyles="--")

    # Data points colored by cluster
    scatter = ax_clusters.scatter(
        X[:, 0], X[:, 1],
        c=labels, cmap="Set1", s=65, alpha=0.85, edgecolors="white"
    )

    # Centroids
    ax_clusters.scatter(
        centroids[:, 0], centroids[:, 1],
        c="#FACC15", marker="X", s=280, edgecolors="black", linewidths=2.0, zorder=8,
        label="Cluster Centroids"
    )

    # Annotate centroids
    for i, c in enumerate(centroids):
        ax_clusters.annotate(
            f"C{i} ({c[0]:.0f}, {c[1]:.0f})",
            xy=(c[0], c[1]),
            xytext=(c[0] + 2, c[1] + 3),
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF08A", edgecolor="#CA8A04")
        )

    ax_clusters.set_title(f"K-Means Clusters & Centroids (Silhouette = {current_silhouette:.3f})", fontsize=13, fontweight="bold", pad=10)
    ax_clusters.set_xlabel("Annual Spending Score (1-100)", fontsize=11)
    ax_clusters.set_ylabel("Store Visit Frequency (Monthly)", fontsize=11)
    ax_clusters.set_xlim(x_min, x_max)
    ax_clusters.set_ylim(y_min, y_max)
    ax_clusters.grid(True, linestyle="--", alpha=0.4)
    ax_clusters.legend(loc="upper left", framealpha=0.9)

    # Subplot 2: Silhouette Score & Elbow Curve
    color_inertia = "#2563EB"
    color_sil = "#DC2626"

    ax_metrics.plot(k_range, inertias, marker="o", color=color_inertia, linewidth=2.2, label="Inertia (Elbow)")
    ax_metrics.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax_metrics.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_inertia, fontsize=11)
    ax_metrics.tick_params(axis="y", labelcolor=color_inertia)
    ax_metrics.grid(True, linestyle="--", alpha=0.4)

    # Twin axis for silhouette score
    ax_sil = ax_metrics.twinx()
    ax_sil.plot(k_range, silhouettes, marker="s", color=color_sil, linewidth=2.2, linestyle="--", label="Silhouette Score")
    ax_sil.set_ylabel("Silhouette Score (Quality of Clustering)", color=color_sil, fontsize=11)
    ax_sil.tick_params(axis="y", labelcolor=color_sil)

    # Highlight chosen k
    chosen_idx = k_range.index(n_clusters)
    ax_metrics.axvline(n_clusters, color="#10B981", linestyle=":", linewidth=2, label=f"Selected k={n_clusters}")

    ax_metrics.set_title("Optimal k Evaluation: Elbow Method & Silhouette Score", fontsize=13, fontweight="bold", pad=10)

    # Combine legends
    lines_1, labels_1 = ax_metrics.get_legend_handles_labels()
    lines_2, labels_2 = ax_sil.get_legend_handles_labels()
    ax_metrics.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "kmeans.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "centroids": centroids,
        "silhouette_score": current_silhouette,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_kmeans()