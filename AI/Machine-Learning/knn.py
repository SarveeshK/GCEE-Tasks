"""
K-Nearest Neighbors (KNN) Concept & Visualization
Distance-based classification with Decision Boundary and Neighbor Linkage / Voting Radius.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_knn(show_plot: bool = True, save_dir: str = "plots", n_neighbors: int = 5):
    print("=" * 60)
    print(f" [K-NEAREST NEIGHBORS (KNN)] Classification (k = {n_neighbors})")
    print("=" * 60)

    # 1. Dataset Generation: Two distinct clusters with slight overlap
    np.random.seed(42)
    n_per_class = 40

    class0_x = np.random.normal(loc=3.0, scale=1.1, size=n_per_class)
    class0_y = np.random.normal(loc=3.0, scale=1.1, size=n_per_class)
    X_0 = np.column_stack((class0_x, class0_y))
    y_0 = np.zeros(n_per_class, dtype=int)

    class1_x = np.random.normal(loc=6.5, scale=1.2, size=n_per_class)
    class1_y = np.random.normal(loc=6.5, scale=1.2, size=n_per_class)
    X_1 = np.column_stack((class1_x, class1_y))
    y_1 = np.ones(n_per_class, dtype=int)

    X = np.vstack((X_0, X_1))
    y = np.concatenate((y_0, y_1))

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Model Training (KNN with k=5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform")
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 5. Query point and nearest neighbors inspection
    query_point = np.array([[5.0, 4.8]])
    query_pred = model.predict(query_point)[0]
    query_prob = model.predict_proba(query_point)[0]

    # Find the k nearest neighbors to the query point
    distances, neighbor_indices = model.kneighbors(query_point)
    neighbor_pts = X_train[neighbor_indices[0]]
    neighbor_labels = y_train[neighbor_indices[0]]

    print(f"KNN Accuracy on Test Set: {acc * 100:.1f}%")
    print(f"Query Point: X1 = {query_point[0, 0]}, X2 = {query_point[0, 1]}")
    print(f"Predicted Class: Class {query_pred} (P(Class 0)={query_prob[0]:.0%}, P(Class 1)={query_prob[1]:.0%})")
    print(f"Nearest {n_neighbors} Neighbors details:")
    for idx, (dist, lbl, pt) in enumerate(zip(distances[0], neighbor_labels, neighbor_pts), 1):
        print(f"  Neighbor {idx}: Coords=({pt[0]:.2f}, {pt[1]:.2f}), Distance={dist:.2f}, Class={lbl}")
    print()

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Mesh grid for decision surface
    x_min, x_max = X[:, 0].min() - 1.2, X[:, 0].max() + 1.2
    y_min, y_max = X[:, 1].min() - 1.2, X[:, 1].max() + 1.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.22, cmap="bwr")
    ax.contour(xx, yy, Z, colors="#374151", linewidths=1.2, linestyles="--")

    # Scatter training points
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="#2563EB", s=60, alpha=0.75, label="Train Class 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="#DC2626", s=60, alpha=0.75, label="Train Class 1")

    # Scatter test points
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="#1D4ED8", marker="s", s=80, edgecolors="black", label="Test Class 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="#B91C1C", marker="s", s=80, edgecolors="black", label="Test Class 1")

    # Connect query point to its k nearest neighbors
    max_dist = distances[0][-1]
    radius_circle = plt.Circle(
        (query_point[0, 0], query_point[0, 1]),
        max_dist,
        color="#F59E0B",
        fill=True,
        alpha=0.12,
        linestyle="--",
        linewidth=1.5
    )
    ax.add_patch(radius_circle)

    for n_pt, n_lbl in zip(neighbor_pts, neighbor_labels):
        line_color = "#DC2626" if n_lbl == 1 else "#2563EB"
        ax.plot([query_point[0, 0], n_pt[0]], [query_point[0, 1], n_pt[1]], color=line_color, linestyle=":", linewidth=1.8)
        # Highlight neighbor circles
        ax.scatter(n_pt[0], n_pt[1], facecolors="none", edgecolors="#10B981", s=130, linewidths=2.2, zorder=5)

    # Plot query point
    ax.scatter(
        query_point[0, 0], query_point[0, 1],
        color="#F59E0B", s=250, marker="*", edgecolor="black", zorder=7,
        label=f"Query Point ({query_point[0,0]}, {query_point[0,1]})"
    )

    class0_votes = np.sum(neighbor_labels == 0)
    class1_votes = np.sum(neighbor_labels == 1)

    ax.annotate(
        f"Query Point\nPred: Class {query_pred}\nVotes: {class0_votes} vs {class1_votes}",
        xy=(query_point[0, 0], query_point[0, 1]),
        xytext=(query_point[0, 0] + 0.6, query_point[0, 1] - 1.2),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B")
    )

    # Cosmetics
    ax.set_title(f"K-Nearest Neighbors (k={n_neighbors}): Decision Boundary & Nearest Neighbor Links", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature 1 (X1)", fontsize=11)
    ax.set_ylabel("Feature 2 (X2)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legend & text
    info_box = f"Neighborhood Voting:\n- k = {n_neighbors}\n- Class 0 Votes: {class0_votes}\n- Class 1 Votes: {class1_votes}\n- Test Acc = {acc*100:.1f}%\nGreen rings = k-NN members"
    ax.text(
        0.03, 0.95, info_box,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.95)
    )

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "knn.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "accuracy": acc,
        "neighbors": neighbor_pts,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_knn()