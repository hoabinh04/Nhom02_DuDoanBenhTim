"""
src/data/cleaner.py — Xử lý thiếu, outlier, encoding cơ bản, cân bằng lớp.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src import load_params, get_path


# ============================================================
# 1. Xử lý giá trị thiếu
# ============================================================
def handle_missing(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Biến số → điền MEDIAN; biến phân loại → điền MODE.
    """
    df = df.copy()
    num_cols = [c for c in params["data"]["numerical_cols"] if c in df.columns]
    cat_cols = [c for c in params["data"]["categorical_cols"] if c in df.columns]

    if num_cols:
        imp = SimpleImputer(strategy="median")
        df[num_cols] = imp.fit_transform(df[num_cols])
    if cat_cols:
        imp = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp.fit_transform(df[cat_cols])

    print(f"[CLEANER] Xử lý missing xong. Còn thiếu: {df.isnull().sum().sum()}")
    return df


# ============================================================
# 2. Nhị phân hóa target
# ============================================================
def binarize_target(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """0 → 0 (không bệnh), 1-4 → 1 (có bệnh)."""
    df = df.copy()
    target = params["data"]["target_col"]
    if params["data"]["binarize_target"]:
        df[target] = (df[target] > 0).astype(int)
        print(f"[CLEANER] Nhị phân hóa target: {df[target].value_counts().to_dict()}")
    return df


# ============================================================
# 3. Mã hóa biến phân loại (Label Encoding)
# ============================================================
def encode_categorical(df: pd.DataFrame, params: dict) -> tuple:
    """Label-encode tất cả biến phân loại. Trả về (df, encoders)."""
    df = df.copy()
    cat_cols = [c for c in params["data"]["categorical_cols"] if c in df.columns]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    print(f"[CLEANER] Mã hóa {len(cat_cols)} biến phân loại")
    return df, encoders


# ============================================================
# 4. Tách train / test
# ============================================================
def split_data(df: pd.DataFrame, params: dict) -> tuple:
    """
    Loại ID & dataset, tách stratified train/test.
    Trả về (X_train, X_test, y_train, y_test).
    """
    df = df.copy()
    drop_cols = [c for c in params["data"]["drop_cols"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    target = params["data"]["target_col"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y,
    )
    print(f"[CLEANER] Tách train/test: Train={len(X_train)}, Test={len(X_test)}")
    print(f"  Train: {y_train.value_counts().to_dict()}")
    print(f"  Test:  {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 5. Chuẩn hóa biến số (StandardScaler)
# ============================================================
def scale_numerical(df: pd.DataFrame, params: dict, scaler=None) -> tuple:
    """Z-score scaling. Trả về (df_scaled, scaler)."""
    df = df.copy()
    num_cols = [c for c in params["data"]["numerical_cols"] if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    print(f"[CLEANER] Chuẩn hóa {len(num_cols)} biến số (StandardScaler)")
    return df, scaler


# ============================================================
# 6. Cân bằng lớp (SMOTE)
# ============================================================
def balance_classes(X: pd.DataFrame, y: pd.Series, params: dict) -> tuple:
    """SMOTE over-sampling trên tập train."""
    sm = SMOTE(random_state=params["seed"])
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[CLEANER] SMOTE: {len(X)} → {len(X_res)} mẫu")
    print(f"  Phân bố sau: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ============================================================
# PIPELINE TIỀN XỬ LÝ TỔNG HỢP
# ============================================================
def run_cleaning_pipeline(df: pd.DataFrame, params: dict = None) -> dict:
    """
    Chạy toàn bộ tiền xử lý theo thứ tự.
    Trả về dict chứa mọi output.
    """
    if params is None:
        params = load_params()

    print("\n" + "=" * 60)
    print("TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # 1) Missing
    df_clean = handle_missing(df, params)
    # 2) Nhị phân hóa target
    df_clean = binarize_target(df_clean, params)
    # 3) Encode
    df_encoded, encoders = encode_categorical(df_clean, params)
    # 4) Split
    X_train, X_test, y_train, y_test = split_data(df_encoded, params)
    # 5) Scale (fit trên train, transform test)
    X_train_sc, scaler = scale_numerical(X_train, params)
    X_test_sc, _ = scale_numerical(X_test, params, scaler=scaler)
    # 6) SMOTE
    X_train_bal, y_train_bal = balance_classes(X_train_sc, y_train, params)

    # Lưu processed
    proc_dir = get_path(params["paths"]["processed_dir"])
    X_train_sc.to_csv(f"{proc_dir}/X_train.csv", index=False)
    X_test_sc.to_csv(f"{proc_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{proc_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{proc_dir}/y_test.csv", index=False)
    print(f"[CLEANER] Đã lưu dữ liệu processed → {proc_dir}")

    return {
        "df_clean": df_clean,
        "df_encoded": df_encoded,
        "encoders": encoders,
        "scaler": scaler,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }
