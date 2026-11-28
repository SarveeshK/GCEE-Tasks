"""
Linear Regression Concept & Visualization
Predicting continuous target (Exam Score) from continuous feature (Study Hours).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def run_linear_regression(show_plot: bool = True, save_dir: str = "plots"):
    print("=" * 60)
    print(" [LINEAR REGRESSION] Study Hours vs. Exam Score")
    print("=" * 60)

    # 1. Dataset generation (Realistic synthetic data with noise)
    np.random.seed(42)
    study_hours = np.array([
        1.0, 1.5, 2.0, 2.3, 2.8, 3.1, 3.5, 3.8, 4.0, 4.2,
        4.7, 5.0, 5.2, 5.5, 5.8, 6.0, 6.4, 6.8, 7.0, 7.3,
        7.7, 8.0, 8.2, 8.5, 8.9, 9.0, 9.2, 9.5, 9.7, 10.0
    ]).reshape(-1, 1)

    # True relationship: Score = 7.5 * Hours + 20 + Gaussian Noise
    noise = np.random.normal(0, 3.5, size=study_hours.shape[0])
    exam_scores = np.clip(7.5 * study_hours.flatten() + 20 + noise, 0, 100)

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        study_hours, exam_scores, test_size=0.25, random_state=42
    )

    # 3. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    slope = model.coef_[0]
    intercept = model.intercept_

    # 4. Evaluation
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # 5. Prediction on new student
    new_hours = np.array([[6.5]])
    new_prediction = model.predict(new_hours)[0]

    print(f"Model Equation: Score = {slope:.2f} * (Hours) + {intercept:.2f}")
    print(f"R^2 Score on Test Set: {r2:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Prediction for a student studying {new_hours[0][0]} hrs: {new_prediction:.2f} marks\n")

    # 6. Visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter actual data
    ax.scatter(X_train, y_train, color="#2563EB", label="Training Data (75%)", s=65, alpha=0.85, edgecolors="none")
    ax.scatter(X_test, y_test, color="#0D9488", label="Test Data (25%)", s=80, marker="s", edgecolors="black", linewidths=0.5)

    # Regression Line
    x_line = np.linspace(0.5, 10.5, 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#DC2626", linewidth=2.5, label=f"Fit Line: y = {slope:.2f}x + {intercept:.2f}")

    # Residuals for test points
    for x_val, y_val, y_hat in zip(X_test.flatten(), y_test, y_pred_test):
        ax.plot([x_val, x_val], [y_val, y_hat], color="#9CA3AF", linestyle="--", linewidth=1.2)

    # Highlight new prediction
    ax.scatter(new_hours[0][0], new_prediction, color="#F59E0B", s=200, marker="*", zorder=5, label=f"Query ({new_hours[0][0]} hrs -> {new_prediction:.1f} marks)")
    ax.annotate(
        f"Query: {new_hours[0][0]} hrs\nPred: {new_prediction:.1f}",
        xy=(new_hours[0][0], new_prediction),
        xytext=(new_hours[0][0] + 0.4, new_prediction - 10),
        arrowprops=dict(facecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9)
    )

    # Cosmetics
    ax.set_title("Linear Regression: Study Hours vs. Exam Score", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Study Hours per Day", fontsize=12)
    ax.set_ylabel("Exam Score (0-100)", fontsize=12)
    ax.set_xlim(0, 11)
    ax.set_ylim(15, 105)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Metrics Box
    stats_text = f"Model Metrics:\n- R^2 = {r2:.3f}\n- MSE = {mse:.2f}\n- MAE = {mae:.2f}\n- Slope = {slope:.2f}\n- Intercept = {intercept:.2f}"
    ax.text(
        0.03, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.95)
    )

    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "linear_regression.png")
    fig.savefig(save_path, dpi=300)
    print(f"[SAVED] Visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "model": model,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mse": mse,
        "plot_path": save_path
    }


if __name__ == "__main__":
    run_linear_regression()