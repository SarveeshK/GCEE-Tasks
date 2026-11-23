"""
Machine Learning Concepts Master Runner
Interactive and batch execution of all 7 machine learning algorithms with automated visualization generation.
"""

import sys
import os
import argparse
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from decision_tree import run_decision_tree
from knn import run_knn
from naive_bayes import run_naive_bayes
from svm import run_svm
from kmeans import run_kmeans

CONCEPTS = {
    "1": ("Linear Regression", run_linear_regression),
    "2": ("Logistic Regression", run_logistic_regression),
    "3": ("Decision Tree", run_decision_tree),
    "4": ("K-Nearest Neighbors (KNN)", run_knn),
    "5": ("Gaussian Naive Bayes", run_naive_bayes),
    "6": ("Support Vector Machine (SVM)", run_svm),
    "7": ("K-Means Clustering", run_kmeans),
}


def print_banner():
    print("=" * 65)
    print("       MACHINE LEARNING CONCEPTS & VISUALIZATIONS SUITE")
    print("=" * 65)
    print("All visualizations are automatically generated & saved to: ./plots/")
    print("-" * 65)


def run_all(show_plots: bool = False, save_dir: str = "plots"):
    print_banner()
    print("Executing all 7 algorithms and generating publication-ready plots...\n")
    results = {}
    for key, (name, func) in CONCEPTS.items():
        print(f"\n>>> [{key}/7] Running {name}...")
        results[name] = func(show_plot=show_plots, save_dir=save_dir)

    print("\n" + "=" * 65)
    print(" ALL 7 CONCEPTS EXECUTED SUCCESSFULLY!")
    print("=" * 65)
    print(f"Generated visualizations saved in folder: '{os.path.abspath(save_dir)}':")
    for name, res in results.items():
        if isinstance(res, dict) and "plot_path" in res:
            print(f"  * {name:<30} -> {res['plot_path']}")
    print("-" * 65)


def interactive_menu():
    print_banner()
    print("Select an algorithm to run:")
    for key, (name, _) in CONCEPTS.items():
        print(f"  [{key}] {name}")
    print("  [8] Run All Concepts (Batch Mode)")
    print("  [q] Quit")
    print("-" * 65)

    choice = input("Enter choice (1-8 or q): ").strip().lower()
    if choice == "q":
        print("Goodbye!")
        return
    elif choice == "8":
        run_all(show_plots=True)
    elif choice in CONCEPTS:
        name, func = CONCEPTS[choice]
        print(f"\nRunning {name}...")
        func(show_plot=True)
    else:
        print("Invalid choice. Running all concepts by default...")
        run_all(show_plots=True)


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Concepts Runner & Visualizer")
    parser.add_argument(
        "--all", action="store_true", help="Run all 7 algorithms in sequence and save plots"
    )
    parser.add_argument(
        "--algo", type=str, choices=["linear", "logistic", "tree", "knn", "nb", "svm", "kmeans"],
        help="Run a specific algorithm"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display GUI plot windows (headless / save only)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="plots", help="Directory where plot images are saved"
    )

    args = parser.parse_args()
    show_plots = not args.no_show

    algo_mapping = {
        "linear": run_linear_regression,
        "logistic": run_logistic_regression,
        "tree": run_decision_tree,
        "knn": run_knn,
        "nb": run_naive_bayes,
        "svm": run_svm,
        "kmeans": run_kmeans,
    }

    if args.all:
        run_all(show_plots=show_plots, save_dir=args.save_dir)
    elif args.algo:
        algo_mapping[args.algo](show_plot=show_plots, save_dir=args.save_dir)
    else:
        # If run directly with no CLI flags and stdin is attached, launch interactive menu
        # If non-interactive, run all
        if sys.stdin.isatty():
            interactive_menu()
        else:
            run_all(show_plots=False, save_dir=args.save_dir)


if __name__ == "__main__":
    main()