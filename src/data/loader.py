"""
src/data/loader.py — Đọc dữ liệu và kiểm tra schema.
"""

import pandas as pd
from src import load_params, get_path


def load_raw_data(params: dict = None) -> pd.DataFrame:
    """Đọc file CSV gốc từ data/raw/ và trả về DataFrame."""
    if params is None:
        params = load_params()
    path = get_path(params["paths"]["raw_data"])
    df = pd.read_csv(path)
    print(f"[LOADER] Đọc thành công: {df.shape[0]} dòng × {df.shape[1]} cột")
    return df


def validate_schema(df: pd.DataFrame, params: dict = None) -> bool:
    """Kiểm tra schema: các cột bắt buộc có tồn tại không."""
    if params is None:
        params = load_params()
    required = (
        params["data"]["categorical_cols"]
        + params["data"]["numerical_cols"]
        + [params["data"]["target_col"]]
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"[LOADER] ⚠ Thiếu cột: {missing_cols}")
        return False
    print("[LOADER] ✓ Schema hợp lệ")
    return True


def inspect_data(df: pd.DataFrame, params: dict = None) -> dict:
    """
    Kiểm tra tổng quan:
    - Kích thước, kiểu dữ liệu
    - Giá trị thiếu
    - Phân bố target
    """
    if params is None:
        params = load_params()
    target = params["data"]["target_col"]
    info = {"shape": df.shape}

    print(f"\n{'='*60}")
    print("TỔNG QUAN DỮ LIỆU")
    print(f"{'='*60}")
    print(f"Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột\n")
    print("Kiểu dữ liệu:")
    print(df.dtypes.to_string())

    # Giá trị thiếu
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Số thiếu": missing, "Tỷ lệ (%)": missing_pct})
    missing_df = missing_df[missing_df["Số thiếu"] > 0].sort_values("Tỷ lệ (%)", ascending=False)
    info["missing"] = missing_df

    if len(missing_df) > 0:
        print(f"\nGiá trị thiếu:")
        print(missing_df.to_string())
    else:
        print(f"\n✓ Không có giá trị thiếu.")

    # Phân bố target
    if target in df.columns:
        dist = df[target].value_counts().sort_index()
        info["target_distribution"] = dist
        print(f"\nPhân bố target (`{target}`):")
        for val, cnt in dist.items():
            print(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print(f"\nThống kê mô tả:")
    print(df.describe().round(2).to_string())
    return info


def load_and_inspect(params: dict = None) -> tuple:
    """Đọc + kiểm tra dữ liệu. Trả về (df, info)."""
    if params is None:
        params = load_params()
    df = load_raw_data(params)
    validate_schema(df, params)
    info = inspect_data(df, params)
    return df, info
