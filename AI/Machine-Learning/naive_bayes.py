"""
Gaussian Naive Bayes Concept & Visualization
Probabilistic classification with Gaussian likelihood estimation, posterior probability heatmap, and decision boundary.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_naive_bayes(show_plot: bool = True, save_dir: str = "plots"):
    print("=" * 60)
    print(" [GAUSSIAN NAIVE BAYES] Diagnostic Classification")
    print("=" * 60)

    # 1. Dataset Generation: Two continuous health biomarkers (Glucose vs Blood Pressure index)
    np.random.seed(42)
    n_samples = 45

    # Class 0: Healthy (Lower glucose, normal BP)
    g_healthy = np.random.normal(loc=90, scale=12, size=n_samples)
    bp_healthy = np.random.normal(loc=72, scale=8, size=n_samples)
    X_0 = np.column_stack((g_healthy, bp_healthy))
    y_0 = np.zeros(n_samples, dtype=int)

    # Class 1: At-Risk (Elevated glucose, elevated BP)
    g_risk = np.random.normal(loc=135, scale=18, size=n_samples)
    bp_risk = np.random.normal(loc=88, scale=10, size=n_samples)
    X_1 = np.column_stack((g_risk, bp_risk))
    y_1 = np.ones(n_samples, dtype=int)

    X = np.vstack((X_0, X_1))
    y = np.concatenate((y_0, y_1))

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Model Training
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Class distributions learned
    means = model.theta_
    variances = model.var_
    priors = model.class_prior_

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 5. Query Patient
    query_patient = np.array([[112.0, 80.0]])
    query_pred = model.predict(query_patient)[0]
    query_prob = model.predict_proba(query_patient)[0]

    print(f"Class Priors: P(Healthy) = {priors[0]:.2f}, P(At-Risk) = {priors[1]:.2f}")
    print(f"Learned Means [Glucose, BP]:")
    print(f"  Healthy: [{means[0, 0]:.1f}, {means[0, 1]:.1f}]")
    print(f"  At-Risk: [{means[1, 0]:.1f}, {means[1, 1]:.1f}]")
    print(f"Test Accuracy: {acc * 100:.1f}%")
    print(f"Query Patient: Glucose = {query_patient[0, 0]}, BP = {query_patient[0, 1]}")
    print(f"  -> P(Healthy) = {query_prob[0]:.1%}, P(At-Risk) = {query_prob[1]:.1%}")
    print(f"  -> Final Diagnosis: {'At-Risk (1)' if query_pred == 1 else 'Healthy (0)'}\n")

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))

    x_min, x_max = 60, 180
    y_min, y_max = 50, 115
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    grid_coords = np.c_[xx.ravel(), yy.ravel()]

    # Predict posterior probability for At-Risk class across grid
    Z_prob = model.predict_proba(grid_coords)[:, 1].reshape(xx.shape)

    # Probability contour surface
    cf = ax.contourf(xx, yy, Z_prob, levels=25, cmap="coolwarm", alpha=0.45)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Posterior Probability P(At-Risk | X)", fontsize=10)

    # Decision boundary at P = 0.5
    cs = ax.contour(xx, yy, Z_prob, levels=[0.5], colors=["#111827"], linewidths=2.5, linestyles="--")
    ax.clabel(cs, inline=True, fmt="P = 0.5 Boundary", fontsize=9)

    # Scatter actual patients
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="#2563EB", s=65, alpha=0.85, label="Train: Healthy")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="#DC2626", s=65, alpha=0.85, label="Train: At-Risk")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="#1E40AF", marker="s", s=80, edgecolors="black", label="Test: Healthy")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="#991B1B", marker="s", s=80, edgecolors="black", label="Test: At-Risk")

    # Mark Gaussian Class Centroids
    ax.scatter(means[0, 0], means[0, 1], color="#1E3A8A", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=6, label=f"Mean Healthy ({means[0,0]:.0f}, {means[0,1]:.0f})")
    ax.scatter(means[1, 0], means[1, 1], color="#7F1D1D", marker="X", s=200, edgecolors="white", linewidths=1.5, zorder=6, label=f"Mean At-Risk ({means[1,0]:.0f}, {means[1,1]:.0f})")

    # Highlight Query Patient
    ax.scatter(
        query_patient[0, 0], query_patient[0, 1],
        color="#F59E0B", s=240, marker="*", edgecolor="black", zorder=7,
        label=f"Query Patient ({query_patient[0,0]:.0f}, {query_patient[0,1]:.0f})"
    )

    ax.annotate(
        f"Query Patient\nP(At-Risk): {query_prob[1]:.1%}\nPred: {'At-Risk' if query_pred==1 else 'Healthy'}",
        xy=(query_patient[0, 0], query_patient[0, 1]),
        xytext=(query_patient[0, 0] - 25, query_patient[0, 1] + 12),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B")
    )

    # Cosmetics
    ax.set_title("Gaussian Naive Bayes: Posterior Probability Surface & Decision Boundary", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Fasting Glucose Level (mg/dL)", fontsize=11)
    ax.set_ylabel("Blood Pressure Index (mm Hg)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Stats card
    info_card = f"Model Specs:\n- Prior Healthy: {priors[0]:.2f}\n- Prior At-Risk: {priors[1]:.2f}\n- Test Accuracy: {acc*100:.1f}%\n- Decision at P(y=1) = 0.5"
    ax.text(
        0.03, 0.95, info_card,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.95)
    )

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "naive_bayes.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "accuracy": acc,
        "means": means,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_naive_bayes()