"""
src/models/supervised.py — Huấn luyện & tinh chỉnh mô hình phân lớp.

Mô hình: LR, DT, RF, SVM, KNN, GradientBoosting, XGBoost
Ghi rõ: hyperparams, thời gian train, thiết lập thực nghiệm.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

from src import load_params, get_path


def _build_models(params: dict) -> dict:
    """Tạo dict tên → (model, param_grid)."""
    seed = params["seed"]
    hp = params["hyperparams"]

    _all = {
        "LogisticRegression": (
            LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced"),
            hp.get("LogisticRegression", {}),
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
            hp.get("DecisionTree", {}),
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced"),
            hp.get("RandomForest", {}),
        ),
        "SVM": (
            SVC(random_state=seed, probability=True, class_weight="balanced"),
            hp.get("SVM", {}),
        ),
        "KNN": (
            KNeighborsClassifier(),
            hp.get("KNN", {}),
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=seed),
            hp.get("GradientBoosting", {}),
        ),
        "XGBoost": (
            XGBClassifier(random_state=seed, eval_metric="logloss"),
            hp.get("XGBoost", {}),
        ),
    }
    names = params["model_names"]
    return {n: _all[n] for n in names if n in _all}


def train_all(X_train, y_train, X_test, y_test, params: dict = None) -> dict:
    """
    GridSearchCV trên từng mô hình. Ghi rõ params & thời gian.
    Trả về dict name → {model, best_params, cv_f1, time, ...}.
    """
    if params is None:
        params = load_params()

    cv = params["cv_folds"]
    models_dir = get_path(params["paths"]["models_dir"])
    tables_dir = get_path(params["paths"]["tables_dir"])

    print("\n" + "=" * 60)
    print("HUẤN LUYỆN MÔ HÌNH PHÂN LỚP")
    print("=" * 60)

    models = _build_models(params)
    results = []

    for name, (model, grid) in models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        gs = GridSearchCV(model, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0

        best = gs.best_estimator_
        cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="f1")

        print(f"  Best params: {gs.best_params_}")
        print(f"  CV F1: {gs.best_score_:.4f} (±{cv_scores.std():.4f})")
        print(f"  Train time: {elapsed:.2f}s")

        # Lưu mô hình
        with open(os.path.join(models_dir, f"{name}_best.pkl"), "wb") as f:
            pickle.dump(best, f)

        results.append({
            "Mô hình": name,
            "Best Params": str(gs.best_params_),
            "CV F1 (mean)": round(gs.best_score_, 4),
            "CV F1 (std)": round(cv_scores.std(), 4),
            "Thời gian (s)": round(elapsed, 2),
            "model": best,
        })

    # Bảng tổng hợp
    df_res = pd.DataFrame(results)
    df_save = df_res.drop(columns=["model"])
    df_save.to_csv(os.path.join(tables_dir, "model_training_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("BẢNG TỔNG HỢP HUẤN LUYỆN")
    print(f"{'='*60}")
    print(df_save.to_string(index=False))

    return {r["Mô hình"]: r for r in results}
