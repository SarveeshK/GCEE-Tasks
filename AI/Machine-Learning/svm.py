"""
Support Vector Machine (SVM) Concept & Visualization
Maximal margin classification, separating hyperplane, margin boundaries, and support vectors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_svm(show_plot: bool = True, save_dir: str = "plots"):
    print("=" * 60)
    print(" [SUPPORT VECTOR MACHINE (SVM)] Maximum Margin Classifier")
    print("=" * 60)

    # 1. Dataset Generation: Two separable 2D distributions
    np.random.seed(42)
    n_samples = 30

    X_0 = np.random.multivariate_normal(mean=[2.5, 3.0], cov=[[0.7, 0.2], [0.2, 0.7]], size=n_samples)
    y_0 = np.zeros(n_samples, dtype=int)

    X_1 = np.random.multivariate_normal(mean=[6.5, 7.0], cov=[[0.8, 0.15], [0.15, 0.8]], size=n_samples)
    y_1 = np.ones(n_samples, dtype=int)

    X = np.vstack((X_0, X_1))
    y = np.concatenate((y_0, y_1))

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Model Training (Linear Kernel with C=1.0)
    model = SVC(kernel="linear", C=1.0, random_state=42)
    model.fit(X_train, y_train)

    w = model.coef_[0]
    b = model.intercept_[0]
    support_vectors = model.support_vectors_
    margin_width = 2.0 / np.linalg.norm(w)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 5. Query point prediction
    query_point = np.array([[4.8, 4.8]])
    query_pred = model.predict(query_point)[0]
    decision_dist = model.decision_function(query_point)[0]

    print(f"Hyperplane Weight vector: w = [{w[0]:.3f}, {w[1]:.3f}]")
    print(f"Hyperplane Intercept (bias): b = {b:.3f}")
    print(f"Number of Support Vectors: {len(support_vectors)} ({model.n_support_[0]} Class 0, {model.n_support_[1]} Class 1)")
    print(f"Margin Width (2 / ||w||): {margin_width:.3f}")
    print(f"Test Set Accuracy: {acc * 100:.1f}%")
    print(f"Query Point ({query_point[0, 0]}, {query_point[0, 1]}):")
    print(f"  -> Decision Function Score: {decision_dist:.3f}")
    print(f"  -> Predicted Class: Class {query_pred}\n")

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))

    x_min, x_max = X[:, 0].min() - 1.2, X[:, 0].max() + 1.2
    y_min, y_max = X[:, 1].min() - 1.2, X[:, 1].max() + 1.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    # Decision function grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot Decision boundary (Z = 0) and Margins (Z = -1, +1)
    ax.contourf(xx, yy, Z, levels=[-1, 1], colors=["#FEF08A"], alpha=0.35)  # Shaded margin gutter
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#4B5563", "#111827", "#4B5563"], linestyles=["--", "-", "--"], linewidths=[1.8, 2.8, 1.8])

    # Scatter points
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="#2563EB", s=65, alpha=0.8, label="Train: Class 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="#DC2626", s=65, alpha=0.8, label="Train: Class 1")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="#1E40AF", marker="s", s=80, edgecolors="black", label="Test: Class 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="#991B1B", marker="s", s=80, edgecolors="black", label="Test: Class 1")

    # Highlight Support Vectors
    ax.scatter(
        support_vectors[:, 0], support_vectors[:, 1],
        s=180, linewidths=2.2, facecolors="none", edgecolors="#10B981", zorder=5,
        label=f"Support Vectors ({len(support_vectors)} total)"
    )

    # Highlight Query Point
    ax.scatter(
        query_point[0, 0], query_point[0, 1],
        color="#F59E0B", s=250, marker="*", edgecolor="black", zorder=7,
        label=f"Query Point ({query_point[0,0]}, {query_point[0,1]})"
    )

    ax.annotate(
        f"Query Point\nPred: Class {query_pred}\nDist to Plane: {decision_dist:.2f}",
        xy=(query_point[0, 0], query_point[0, 1]),
        xytext=(query_point[0, 0] - 1.8, query_point[0, 1] + 1.3),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B")
    )

    # Cosmetics
    ax.set_title("Support Vector Machine: Maximum Margin & Support Vectors", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature 1 (X1)", fontsize=11)
    ax.set_ylabel("Feature 2 (X2)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Info box
    info_box = f"SVM Geometry:\n- Solid line: Hyperplane (w.x + b = 0)\n- Dashed lines: Margins (+/- 1)\n- Yellow band: Margin Gutter\n- Margin Width: {margin_width:.2f}\n- Test Accuracy: {acc*100:.1f}%\n- Green rings: Support Vectors"
    ax.text(
        0.03, 0.95, info_box,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.95)
    )

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "svm.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "accuracy": acc,
        "support_vectors": support_vectors,
        "margin_width": margin_width,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_svm()