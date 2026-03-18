"""
src/models/regression.py — Hồi quy chỉ số (vd: huyết áp) theo yếu tố nguy cơ.

Mô hình: LinearRegression, Ridge, XGBRegressor
Metric:  MAE, RMSE
Kiểm tra: outlier, leakage
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src import load_params, get_path


def prepare_regression_data(df_clean, params: dict = None):
    """
    Chuẩn bị dữ liệu cho bài toán hồi quy.
    Target mặc định: trestbps (huyết áp lúc nghỉ).
    Features: các yếu tố nguy cơ (trừ target hồi quy & id/dataset).
    """
    if params is None:
        params = load_params()

    reg_cfg = params.get("regression", {})
    reg_target = reg_cfg.get("target_col", "trestbps")
    drop_cols = reg_cfg.get("drop_cols", ["id", "dataset", "num"])

    df = df_clean.copy()

    # Drop cols
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if reg_target not in df.columns:
        raise ValueError(f"Cột hồi quy '{reg_target}' không có trong dữ liệu!")

    y = df[reg_target]
    X = df.drop(columns=[reg_target])

    # Encode categoricals nếu còn object
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    print(f"[REGRESSION] Target: {reg_target}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  y stats: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min()}, max={y.max()}")

    return X, y, reg_target


def check_outliers_leakage(X, y, reg_target, params: dict = None):
    """
    Kiểm tra outlier trong target & feature leakage.
    """
    if params is None:
        params = load_params()

    print(f"\n{'─'*50}")
    print("KIỂM TRA OUTLIER & LEAKAGE")
    print(f"{'─'*50}")

    # Outlier detection (IQR)
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = ((y < lower) | (y > upper)).sum()
    print(f"  Target '{reg_target}': IQR=[{Q1:.1f}, {Q3:.1f}], bounds=[{lower:.1f}, {upper:.1f}]")
    print(f"  Outliers: {outliers} / {len(y)} ({outliers/len(y)*100:.1f}%)")

    # Leakage check: correlation quá cao
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\n  Top tương quan với '{reg_target}':")
    for feat, val in corr.head(5).items():
        flag = " ⚠ LEAKAGE?" if val > 0.95 else ""
        print(f"    {feat}: r = {val:.4f}{flag}")

    leaky = corr[corr > 0.95].index.tolist()
    if leaky:
        print(f"\n  ⚠ Cảnh báo: cột {leaky} tương quan > 0.95 → có thể leakage!")
    else:
        print(f"\n  ✓ Không phát hiện leakage rõ ràng.")

    return {"outliers": outliers, "leaky_cols": leaky, "correlations": corr}


def train_regression(X, y, params: dict = None) -> dict:
    """
    Huấn luyện LinearRegression, Ridge, XGBRegressor.
    Cross-validation + ghi thời gian train.
    Trả về dict name → {model, mae, rmse, ...}.
    """
    if params is None:
        params = load_params()

    seed = params["seed"]
    cv = params["cv_folds"]
    reg_cfg = params.get("regression", {})
    hp = reg_cfg.get("hyperparams", {})
    models_dir = get_path(params["paths"]["models_dir"])
    tables_dir = get_path(params["paths"]["tables_dir"])

    print(f"\n{'='*60}")
    print("HUẤN LUYỆN MÔ HÌNH HỒI QUY")
    print(f"{'='*60}")

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=hp.get("Ridge", {}).get("alpha", 1.0), random_state=seed),
        "XGBRegressor": XGBRegressor(
            n_estimators=hp.get("XGBRegressor", {}).get("n_estimators", 200),
            max_depth=hp.get("XGBRegressor", {}).get("max_depth", 5),
            learning_rate=hp.get("XGBRegressor", {}).get("learning_rate", 0.1),
            random_state=seed,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        # Cross-validation
        cv_mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))

        # Fit toàn bộ
        model.fit(X, y)
        elapsed = time.time() - t0
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"  CV MAE: {cv_mae.mean():.4f} (±{cv_mae.std():.4f})")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
        print(f"  Train MAE: {mae:.4f}, Train RMSE: {rmse:.4f}")
        print(f"  Train time: {elapsed:.2f}s")

        # Lưu mô hình
        with open(os.path.join(models_dir, f"reg_{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        results.append({
            "Mô hình": name,
            "CV MAE (mean)": round(cv_mae.mean(), 4),
            "CV MAE (std)": round(cv_mae.std(), 4),
            "CV RMSE (mean)": round(cv_rmse.mean(), 4),
            "CV RMSE (std)": round(cv_rmse.std(), 4),
            "Train MAE": round(mae, 4),
            "Train RMSE": round(rmse, 4),
            "Thời gian (s)": round(elapsed, 2),
            "model": model,
        })

    df_res = pd.DataFrame(results)
    df_save = df_res.drop(columns=["model"])
    df_save.to_csv(os.path.join(tables_dir, "regression_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("BẢNG TỔNG HỢP HỒI QUY")
    print(f"{'='*60}")
    print(df_save.to_string(index=False))

    return {r["Mô hình"]: r for r in results}


def plot_regression_results(reg_results, X, y, params: dict = None):
    """Biểu đồ: Actual vs Predicted + Residual + Feature Importance."""
    if params is None:
        params = load_params()
    fig_dir = get_path(params["paths"]["figures_dir"])

    names = list(reg_results.keys())
    n = len(names)

    # Actual vs Predicted
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        model = reg_results[name]["model"]
        y_pred = model.predict(X)
        ax.scatter(y, y_pred, s=10, alpha=0.5, c="#3498db")
        mn, mx = y.min(), y.max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect")
        ax.set_title(f"{name}\nMAE={reg_results[name]['CV MAE (mean)']:.3f}", fontweight="bold")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.legend()
    plt.suptitle("Actual vs Predicted", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "regression_actual_vs_pred.png"), bbox_inches="tight")
    plt.show()

    # Residual plot for best model
    best_name = min(reg_results, key=lambda n: reg_results[n]["CV MAE (mean)"])
    best_model = reg_results[best_name]["model"]
    y_pred = best_model.predict(X)
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, s=10, alpha=0.5, c="#e74c3c")
    axes[0].axhline(0, color="black", ls="--")
    axes[0].set_title(f"Residual Plot — {best_name}", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", ls="--")
    axes[1].set_title("Phân bố Residual", fontweight="bold")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Tần suất")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "regression_residuals.png"), bbox_inches="tight")
    plt.show()

    # Feature Importance (XGBRegressor)
    if "XGBRegressor" in reg_results:
        xgb = reg_results["XGBRegressor"]["model"]
        imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        imp.tail(10).plot(kind="barh", ax=ax, color="#2ecc71", edgecolor="white")
        ax.set_title("Feature Importance — XGBRegressor", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "regression_feature_importance.png"), bbox_inches="tight")
        plt.show()
