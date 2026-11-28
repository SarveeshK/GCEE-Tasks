"""
Logistic Regression Concept & Visualization
Predicting binary outcome (Fail = 0, Pass = 1) and outputting probabilities using the Sigmoid curve.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def run_logistic_regression(show_plot: bool = True, save_dir: str = "plots"):
    print("=" * 60)
    print(" [LOGISTIC REGRESSION] Pass/Fail Probability vs. Study Hours")
    print("=" * 60)

    # 1. Dataset generation
    np.random.seed(42)
    hours = np.array([
        0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2,
        3.5, 3.7, 3.9, 4.0, 4.2, 4.5, 4.8, 5.0, 5.3, 5.5,
        5.8, 6.0, 6.5, 7.0, 7.5, 8.0
    ]).reshape(-1, 1)

    # Probabilistic label generation around 3.8 hours threshold
    # Ground truth sigmoid: p = 1 / (1 + exp(-2*(x - 3.8)))
    true_prob = 1 / (1 + np.exp(-2.2 * (hours.flatten() - 3.8)))
    y = (np.random.rand(len(hours)) < true_prob).astype(int)

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        hours, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Model Training
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    w1 = model.coef_[0][0]
    b = model.intercept_[0]
    # Decision boundary occurs where w1*x + b = 0 => x = -b / w1
    decision_boundary = -b / w1

    # 4. Evaluation
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    # 5. Query prediction
    query_hours = np.array([[3.2]])
    query_pred = model.predict(query_hours)[0]
    query_prob = model.predict_proba(query_hours)[0]

    print(f"Model Parameters: Weight (w) = {w1:.3f}, Bias (b) = {b:.3f}")
    print(f"Calculated Decision Boundary (P = 0.5): {decision_boundary:.2f} hours")
    print(f"Test Accuracy: {acc * 100:.1f}%")
    print(f"Precision: {prec:.2f} | Recall: {rec:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Query Student ({query_hours[0][0]} hrs):")
    print(f"  -> P(Fail) = {query_prob[0]:.2%}, P(Pass) = {query_prob[1]:.2%}")
    print(f"  -> Predicted Class: {'Pass (1)' if query_pred == 1 else 'Fail (0)'}\n")

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax_main, ax_cm) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.6, 1]}
    )

    # Dense X for smooth sigmoid curve
    x_dense = np.linspace(0.5, 8.5, 300).reshape(-1, 1)
    y_prob_curve = model.predict_proba(x_dense)[:, 1]

    # Plot actual points
    train_fail = X_train[y_train == 0]
    train_pass = X_train[y_train == 1]
    test_fail = X_test[y_test == 0]
    test_pass = X_test[y_test == 1]

    ax_main.scatter(train_fail, np.zeros_like(train_fail), color="#DC2626", s=60, alpha=0.8, label="Train: Fail (0)")
    ax_main.scatter(train_pass, np.ones_like(train_pass), color="#16A34A", s=60, alpha=0.8, label="Train: Pass (1)")
    ax_main.scatter(test_fail, np.zeros_like(test_fail), color="#991B1B", marker="s", s=80, edgecolors="black", label="Test: Fail (0)")
    ax_main.scatter(test_pass, np.ones_like(test_pass), color="#166534", marker="s", s=80, edgecolors="black", label="Test: Pass (1)")

    # Plot Sigmoid Curve
    ax_main.plot(x_dense, y_prob_curve, color="#2563EB", linewidth=3, label="Logistic Sigmoid P(Pass)")

    # Decision Boundary & 0.5 Threshold lines
    ax_main.axvline(decision_boundary, color="#6B7280", linestyle=":", linewidth=2, label=f"Boundary (x = {decision_boundary:.2f}h)")
    ax_main.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1.2, label="Threshold (p = 0.5)")

    # Highlight Query Student
    query_p_pass = query_prob[1]
    ax_main.scatter(query_hours[0][0], query_p_pass, color="#F59E0B", s=220, marker="*", zorder=6, label=f"Query ({query_hours[0][0]}h -> P={query_p_pass:.0%})")
    ax_main.vlines(query_hours[0][0], 0, query_p_pass, colors="#F59E0B", linestyles="--", linewidth=1.5)
    ax_main.hlines(query_p_pass, 0.5, query_hours[0][0], colors="#F59E0B", linestyles="--", linewidth=1.5)

    ax_main.annotate(
        f"Query: {query_hours[0][0]}h\nP(Pass)={query_p_pass:.1%}\nPred: {'Pass' if query_pred==1 else 'Fail'}",
        xy=(query_hours[0][0], query_p_pass),
        xytext=(query_hours[0][0] - 1.7, query_p_pass + 0.15),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B")
    )

    ax_main.set_title("Logistic Regression: Sigmoid Curve & Decision Boundary", fontsize=13, fontweight="bold")
    ax_main.set_xlabel("Study Hours per Day", fontsize=11)
    ax_main.set_ylabel("Predicted Probability P(Pass | Hours)", fontsize=11)
    ax_main.set_ylim(-0.08, 1.08)
    ax_main.set_xlim(0.5, 8.5)
    ax_main.grid(True, linestyle="--", alpha=0.5)
    ax_main.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # Subplot 2: Confusion Matrix Heatmap
    cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Fail (0)", "Pass (1)"])
    ax_cm.set_yticklabels(["Fail (0)", "Pass (1)"])
    ax_cm.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax_cm.set_ylabel("Actual True Label", fontsize=10)
    ax_cm.set_title(f"Test Confusion Matrix\n(Acc = {acc*100:.0f}%)", fontsize=11, fontweight="bold", pad=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "logistic_regression.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "accuracy": acc,
        "decision_boundary": decision_boundary,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_logistic_regression()