"""
src/evaluation/metrics.py — Evaluation metrics for supervised models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def evaluate_all(models_results: dict, X_test, y_test) -> dict:
    """
    Đánh giá tất cả models trên test set.
    Trả về dict name → metrics dict.
    """
    evaluations = {}

    for name, res in models_results.items():
        model = res["model"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        if y_proba is not None:
            metrics["auc"] = roc_auc_score(y_test, y_proba)

        evaluations[name] = metrics
        print(f"\n{name} Test Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        if "auc" in metrics:
            print(f"  AUC: {metrics['auc']:.4f}")

    return evaluations