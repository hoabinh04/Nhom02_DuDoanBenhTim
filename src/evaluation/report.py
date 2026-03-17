"""
src/evaluation/report.py — Generate evaluation reports and comparisons.
"""

import os
import pandas as pd
from src import get_path


def save_comparison(evaluations: dict, params: dict = None):
    """
    Lưu bảng so sánh metrics vào outputs/tables/.
    """
    if params is None:
        from src import load_params
        params = load_params()

    tables_dir = get_path("tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Create comparison DataFrame
    comparison = {}
    for name, metrics in evaluations.items():
        comparison[name] = {
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "AUC": metrics.get("auc", 0)
        }

    df_comp = pd.DataFrame(comparison).T
    comp_path = os.path.join(tables_dir, "model_comparison.csv")
    df_comp.to_csv(comp_path)
    print(f"Saved model comparison to {comp_path}")

    return df_comp


def print_insights(evaluations: dict):
    """
    In ra insights từ evaluation results.
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON INSIGHTS")
    print("="*60)

    # Find best models
    best_acc = max(evaluations.items(), key=lambda x: x[1]["accuracy"])
    best_f1 = max(evaluations.items(), key=lambda x: x[1]["f1"])
    best_auc = max(evaluations.items(), key=lambda x: x[1].get("auc", 0))

    print(f"🏆 Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
    print(f"🏆 Best F1 Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    print(f"🏆 Best AUC: {best_auc[0]} ({best_auc[1].get('auc', 0):.4f})")

    # Recommendations
    print("\n💡 Recommendations:")
    if best_f1[1]['recall'] > 0.8:
        print("  - High recall model recommended for medical diagnosis (catch more positives)")
    elif best_f1[1]['precision'] > 0.8:
        print("  - High precision model recommended when false positives are costly")

    print("="*60)