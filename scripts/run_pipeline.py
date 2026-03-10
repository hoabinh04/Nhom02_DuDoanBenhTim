"""
scripts/run_pipeline.py — Chạy toàn bộ pipeline từ dòng lệnh.

Sử dụng:
    cd C:\\KHMT\\BTL-DATAMINING
    python scripts/run_pipeline.py
"""

import sys
import os

# Đảm bảo import được src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_params
from src.data.loader import load_raw_data, inspect_data
from src.data.cleaner import run_cleaning_pipeline
from src.features.builder import create_features, feature_importance
from src.mining.association import mine_rules
from src.mining.clustering import run_clustering
from src.mining.anomaly import detect_anomalies
from src.models.supervised import train_all
from src.models.semi_supervised import run_semi_supervised
from src.models.regression import (
    prepare_regression_data, check_outliers_leakage,
    train_regression, plot_regression_results,
)
from src.evaluation.metrics import evaluate_all
from src.evaluation.report import save_comparison, print_insights
from src.visualization.plots import run_all_eda_plots, run_all_eval_plots


def main():
    params = load_params()

    print("╔" + "═" * 58 + "╗")
    print("║   PIPELINE DỰ ĐOÁN BỆNH TIM — UCI Heart Disease          ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. Load dữ liệu ──────────────────────────────────
    print("\n[1/9] Đọc dữ liệu thô ...")
    df = load_raw_data(params)
    inspect_data(df)

    # ── 2. EDA plots ─────────────────────────────────────
    print("\n[2/9] EDA — Biểu đồ khám phá ...")
    run_all_eda_plots(df, params)

    # ── 3. Tiền xử lý ───────────────────────────────────
    print("\n[3/9] Tiền xử lý & Feature Engineering ...")
    prep = run_cleaning_pipeline(df, params)
    feat = create_features(prep["df_encoded"], params)
    fi = feature_importance(prep["X_train"], prep["y_train"], params)

    # ── 4. Mining ────────────────────────────────────────
    print("\n[4/9] Khai phá Tri thức ...")
    rules = mine_rules(prep["df_clean"], params)
    clustering_results = run_clustering(prep["X_train"], prep["y_train"], params)
    anomaly_labels = detect_anomalies(prep["X_train"], params)
    n_outliers = (anomaly_labels == -1).sum()
    print(f"  Anomaly: {n_outliers} điểm bất thường / {len(anomaly_labels)}")

    # ── 5. Supervised Modeling ───────────────────────────
    print("\n[5/9] Huấn luyện 7 mô hình phân lớp ...")
    model_results = train_all(
        prep["X_train_bal"], prep["y_train_bal"],
        prep["X_test"], prep["y_test"],
        params,
    )

    # ── 6. Đánh giá ─────────────────────────────────────
    print("\n[6/9] Đánh giá trên tập Test ...")
    comp_df, preds = evaluate_all(model_results, prep["X_test"], prep["y_test"])

    # ── 7. Semi-supervised ───────────────────────────────
    print("\n[7/9] Thực nghiệm bán giám sát ...")
    df_semi = run_semi_supervised(
        prep["X_train"], prep["y_train"],
        prep["X_test"], prep["y_test"],
        params,
    )

    # ── 8. Hồi quy chỉ số ───────────────────────────────
    print("\n[8/9] Hồi quy chỉ số (huyết áp) ...")
    from src.data.cleaner import handle_missing, encode_categorical
    df_reg = handle_missing(df, params)
    df_reg, _ = encode_categorical(df_reg, params)
    X_reg, y_reg, reg_target = prepare_regression_data(df_reg, params)
    check_outliers_leakage(X_reg, y_reg, reg_target, params)
    reg_results = train_regression(X_reg, y_reg, params)
    plot_regression_results(reg_results, X_reg, y_reg, params)

    # ── 9. Report & Viz ─────────────────────────────
    print("\n[9/9] Lưu báo cáo & Biểu đồ đánh giá ...")
    save_comparison(comp_df, params)
    run_all_eval_plots(comp_df, model_results, preds, prep["X_test"], prep["y_test"], params)
    print_insights(comp_df)

    print("\n" + "═" * 60)
    print("PIPELINE HOÀN TẤT!")
    print("═" * 60)
    print("  outputs/figures/  → Biểu đồ")
    print("  outputs/tables/   → Bảng kết quả CSV")
    print("  outputs/models/   → Mô hình .pkl")
    print("  data/processed/   → Dữ liệu đã xử lý")


if __name__ == "__main__":
    main()
