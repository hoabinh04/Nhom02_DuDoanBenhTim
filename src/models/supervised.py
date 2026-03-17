"""
src/models/supervised.py — Supervised learning models for heart disease prediction.

Models: LogisticRegression, RandomForest, XGBoost, SVM
Metrics: Accuracy, Precision, Recall, F1, AUC
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src import load_params, get_path


def prepare_classification_data(df_clean, params: dict = None):
    """
    Chuẩn bị dữ liệu cho bài toán phân loại.
    Target mặc định: target (0: không bệnh, 1: bệnh tim).
    Features: các yếu tố nguy cơ (trừ target & id/dataset).
    """
    if params is None:
        params = load_params()

    clf_cfg = params.get("classification", {})
    clf_target = clf_cfg.get("target_col", "target")
    drop_cols = clf_cfg.get("drop_cols", ["id", "dataset"])

    df = df_clean.copy()

    # Drop cols
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if clf_target not in df.columns:
        raise ValueError(f"Cột phân loại '{clf_target}' không có trong dữ liệu!")

    y = df[clf_target]
    X = df.drop(columns=[clf_target])

    # Encode categoricals nếu còn object
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    print(f"[CLASSIFICATION] Target: {clf_target}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    return X, y, clf_target


def train_all(X, y, params: dict = None) -> dict:
    """
    Huấn luyện tất cả models: LogisticRegression, RandomForest, XGBoost, SVM.
    Sử dụng SMOTE nếu imbalance.
    Cross-validation + ghi thời gian train.
    Trả về dict name → {model, acc, prec, rec, f1, auc, ...}.
    """
    if params is None:
        params = load_params()

    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "SVM": SVC(random_state=42, probability=True)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Check imbalance
    if y.value_counts().min() / y.value_counts().max() < 0.5:
        print("⚠ Dataset imbalance detected. Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"After SMOTE: {y.value_counts().to_dict()}")

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"TRAINING {name}")
        print(f"{'─'*50}")

        start_time = time.time()

        # Cross-validation scores
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        prec_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
        rec_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        train_time = time.time() - start_time

        # Fit full model for saving
        model.fit(X, y)

        results[name] = {
            "model": model,
            "accuracy": acc_scores.mean(),
            "precision": prec_scores.mean(),
            "recall": rec_scores.mean(),
            "f1": f1_scores.mean(),
            "auc": auc_scores.mean(),
            "train_time": train_time,
            "cv_scores": {
                "accuracy": acc_scores,
                "precision": prec_scores,
                "recall": rec_scores,
                "f1": f1_scores,
                "auc": auc_scores
            }
        }

        print(f"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
        print(f"  Precision: {prec_scores.mean():.4f} ± {prec_scores.std():.4f}")
        print(f"  Recall: {rec_scores.mean():.4f} ± {rec_scores.std():.4f}")
        print(f"  F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        print(f"  AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        print(f"  Train time: {train_time:.2f}s")

    return results


def save_models(results: dict, params: dict = None):
    """
    Lưu models đã train vào outputs/models/.
    """
    if params is None:
        params = load_params()

    models_dir = get_path("models")
    os.makedirs(models_dir, exist_ok=True)

    for name, res in results.items():
        model_path = os.path.join(models_dir, f"{name.lower()}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(res["model"], f)
        print(f"Saved {name} to {model_path}")