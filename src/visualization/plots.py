"""
src/visualization/plots.py — Evaluation plots for models.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from src import get_path


def run_all_eval_plots(evaluations: dict, X_test, y_test, params: dict = None):
    """
    Tạo tất cả evaluation plots: confusion matrices, ROC curves, etc.
    """
    if params is None:
        from src import load_params
        params = load_params()

    figures_dir = get_path("figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (name, metrics) in enumerate(evaluations.items()):
        if i >= 4:
            break

        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")

    # ROC Curves
    plt.figure(figsize=(10, 8))
    for name, metrics in evaluations.items():
        if "auc" in metrics:
            # Note: Need actual y_proba, but for now skip detailed ROC
            pass

    # Placeholder
    plt.title('ROC Curves (Placeholder)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

    print("Evaluation plots generated successfully.")