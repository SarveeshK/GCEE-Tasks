"""
Decision Tree Concept & Visualization
Loan Approval classification based on Age and Annual Income, featuring Tree Structure & Decision Surface.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def run_decision_tree(show_plot: bool = True, save_dir: str = "plots"):
    print("=" * 60)
    print(" [DECISION TREE] Loan Approval by Age & Income")
    print("=" * 60)

    # 1. Dataset Generation (Age in years, Income in thousands USD)
    np.random.seed(42)
    n_samples = 80

    age = np.random.uniform(20, 65, n_samples)
    income_k = np.random.uniform(20, 120, n_samples)
    X = np.column_stack((age, income_k))

    # Realistic approval rule with non-linear tree-friendly boundaries:
    # High income (> 60k) or (moderate income > 40k and age > 30)
    approved = (
        (income_k > 65) |
        ((income_k > 40) & (age >= 32)) |
        ((income_k > 85))
    ).astype(int)

    # Add small realistic boundary noise (~5%)
    flip_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    approved[flip_indices] = 1 - approved[flip_indices]
    y = approved

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    feature_names = ["Age (years)", "Income ($k)"]
    class_names = ["Rejected", "Approved"]

    # 3. Model Training
    model = DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    importances = model.feature_importances_

    # 5. Query applicant
    query_applicant = np.array([[40.0, 50.0]])
    query_pred = model.predict(query_applicant)[0]
    query_prob = model.predict_proba(query_applicant)[0]

    print(f"Decision Tree Max Depth: {model.get_depth()} | Leaves: {model.get_n_leaves()}")
    print(f"Test Accuracy: {acc * 100:.1f}%")
    print(f"Feature Importances: Age = {importances[0]:.3f}, Income = {importances[1]:.3f}")
    print(f"Query Applicant: Age = {query_applicant[0][0]}, Income = ${query_applicant[0][1]:.0f}k")
    print(f"  -> Prediction: {class_names[query_pred]} (Prob Approved: {query_prob[1]:.1%})\n")

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(15, 6.5))

    # Subplot 1: Tree Graph Architecture
    ax_tree = fig.add_subplot(1, 2, 1)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax_tree,
        proportion=True
    )
    ax_tree.set_title("Decision Tree Hierarchy & Split Rules", fontsize=13, fontweight="bold", pad=10)

    # Subplot 2: 2D Feature Space Decision Surface
    ax_surface = fig.add_subplot(1, 2, 2)
    x_min, x_max = 18, 68
    y_min, y_max = 15, 125
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    # Decision regions
    ax_surface.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax_surface.contour(xx, yy, Z, colors="black", linewidths=1.2, linestyles="--")

    # Data points
    scatter_train = ax_surface.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train, cmap="coolwarm", edgecolor="white", s=65,
        alpha=0.85, label="Training Applicants"
    )
    scatter_test = ax_surface.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap="coolwarm", marker="s", edgecolor="black", s=75,
        linewidths=0.8, label="Test Applicants"
    )

    # Query point
    ax_surface.scatter(
        query_applicant[0, 0], query_applicant[0, 1],
        color="#F59E0B", s=220, marker="*", edgecolor="black", zorder=6,
        label=f"Query ({query_applicant[0,0]:.0f}y, ${query_applicant[0,1]:.0f}k)"
    )

    ax_surface.annotate(
        f"Query Applicant\nPred: {class_names[query_pred]}",
        xy=(query_applicant[0, 0], query_applicant[0, 1]),
        xytext=(query_applicant[0, 0] - 14, query_applicant[0, 1] + 12),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B")
    )

    ax_surface.set_title(f"Decision Surface in Feature Space (Acc = {acc*100:.0f}%)", fontsize=13, fontweight="bold", pad=10)
    ax_surface.set_xlabel("Applicant Age (years)", fontsize=11)
    ax_surface.set_ylabel("Annual Income ($1,000s)", fontsize=11)
    ax_surface.set_xlim(x_min, x_max)
    ax_surface.set_ylim(y_min, y_max)
    ax_surface.grid(True, linestyle="--", alpha=0.4)
    ax_surface.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "decision_tree.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "accuracy": acc,
        "importances": importances,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_decision_tree()