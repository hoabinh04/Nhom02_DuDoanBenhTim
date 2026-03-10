"""
src/features/builder.py — Feature engineering:
  - Tạo đặc trưng phái sinh
  - Đánh giá tầm quan trọng (Mutual Information)
  - Chọn top-K đặc trưng (ANOVA F)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif


def create_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Tạo các đặc trưng phái sinh cho bài toán bệnh tim."""
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]
        ).astype(float)

    if "age" in df.columns and "trestbps" in df.columns:
        df["age_bp_interaction"] = df["age"] * df["trestbps"]

    if "age" in df.columns and "chol" in df.columns:
        df["chol_age_ratio"] = df["chol"] / (df["age"] + 1)

    if "thalch" in df.columns and "oldpeak" in df.columns:
        df["hr_oldpeak_diff"] = df["thalch"] - df["oldpeak"] * 10

    if "trestbps" in df.columns and "thalch" in df.columns:
        df["bp_hr_ratio"] = df["trestbps"] / (df["thalch"] + 1)

    print(f"[FEATURE] Tạo đặc trưng mới: {df.shape[1]} cột tổng cộng")
    return df


def feature_importance(X: pd.DataFrame, y: pd.Series, params: dict = None) -> pd.DataFrame:
    """Tính Mutual Information cho từng đặc trưng."""
    mi = mutual_info_classif(X, y, random_state=42)
    result = pd.DataFrame({
        "Đặc trưng": X.columns,
        "MI Score": mi
    }).sort_values("MI Score", ascending=False).reset_index(drop=True)
    print("[FEATURE] Tầm quan trọng (MI):")
    print(result.to_string(index=False))
    return result


def select_top_k(X: pd.DataFrame, y: pd.Series, k: int = 10) -> tuple:
    """Chọn top-K đặc trưng bằng ANOVA F-value."""
    k = min(k, X.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    mask = selector.get_support()
    cols = X.columns[mask].tolist()
    print(f"[FEATURE] Top-{k}: {cols}")
    return pd.DataFrame(X_sel, columns=cols, index=X.index), cols
