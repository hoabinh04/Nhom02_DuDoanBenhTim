"""
src/evaluation/report.py — Tổng hợp bảng kết quả, insight, khuyến nghị.
"""

import os
import pandas as pd

from src import load_params, get_path


def save_comparison(comp_df: pd.DataFrame, params: dict = None):
    """Lưu bảng so sánh mô hình ra CSV."""
    if params is None:
        params = load_params()
    path = os.path.join(get_path(params["paths"]["tables_dir"]), "model_comparison.csv")
    comp_df.to_csv(path, index=False)
    print(f"[REPORT] Đã lưu: {path}")


def print_insights(comp_df: pd.DataFrame):
    """In 5+ insight hành động (actionable) dựa trên kết quả."""
    print(f"\n{'='*60}")
    print("INSIGHT & KHUYẾN NGHỊ HÀNH ĐỘNG")
    print(f"{'='*60}")

    best = comp_df.iloc[0]
    best_recall = comp_df.loc[comp_df["Recall"].idxmax()]
    best_prec = comp_df.loc[comp_df["Precision"].idxmax()]
    worst = comp_df.iloc[-1]

    # Insight 1: Mô hình tốt nhất
    print(f"\n  [1] MÔ HÌNH TỐT NHẤT (F1):")
    print(f"      → {best['Mô hình']} — F1={best['F1-Score']}, AUC={best.get('AUC-ROC', 'N/A')}")
    print(f"      Hành động: Dùng mô hình này làm mô hình chính (primary model) cho triển khai")

    # Insight 2: Ưu tiên Recall cho y tế
    print(f"\n  [2] ƯU TIÊN RECALL (quan trọng y tế):")
    print(f"      → {best_recall['Mô hình']} có Recall={best_recall['Recall']} (cao nhất)")
    print(f"      Hành động: Trong sàng lọc ban đầu, ưu tiên mô hình Recall cao để KHÔNG BỎ SÓT bệnh")
    print(f"      → False Negative (bỏ sót bệnh nhân) nguy hiểm hơn False Positive (báo nhầm)")

    # Insight 3: Trade-off Precision/Recall
    print(f"\n  [3] TRADE-OFF PRECISION / RECALL:")
    print(f"      → Precision cao nhất: {best_prec['Mô hình']} (P={best_prec['Precision']})")
    print(f"      → Recall cao nhất:    {best_recall['Mô hình']} (R={best_recall['Recall']})")
    print(f"      Hành động: Nếu chi phí xét nghiệm cao → chọn mô hình P cao;")
    print(f"      Nếu ưu tiên an toàn bệnh nhân → chọn mô hình R cao")

    # Insight 4: Mô hình yếu nhất
    print(f"\n  [4] MÔ HÌNH CẦN CẢI THIỆN:")
    print(f"      → {worst['Mô hình']} — F1={worst['F1-Score']} (thấp nhất)")
    gap = best["F1-Score"] - worst["F1-Score"]
    print(f"      Khoảng cách với mô hình tốt nhất: {gap:.4f}")
    print(f"      Hành động: Thử tinh chỉnh hyperparameter sâu hơn hoặc loại khỏi pipeline")

    # Insight 5: Ensemble recommendation
    print(f"\n  [5] ĐỀ XUẤT ENSEMBLE:")
    top3 = comp_df.head(3)["Mô hình"].tolist()
    print(f"      → Kết hợp top-3: {', '.join(top3)}")
    print(f"      Hành động: Voting/Stacking ensemble có thể cải thiện thêm 1-3% F1")

    # Insight 6 (nếu có PR-AUC)
    if "PR-AUC" in comp_df.columns:
        best_pr = comp_df.loc[comp_df["PR-AUC"].idxmax()]
        print(f"\n  [6] PR-AUC (tốt hơn ROC-AUC cho imbalanced):")
        print(f"      → {best_pr['Mô hình']} — PR-AUC={best_pr['PR-AUC']}")
        print(f"      Hành động: Dùng PR-AUC làm metric chính khi class imbalanced")

    # Insight 7: Thu thập dữ liệu
    print(f"\n  [7] THU THẬP DỮ LIỆU:")
    print(f"      → Dữ liệu hiện tại: 920 mẫu từ 4 bệnh viện")
    print(f"      Hành động: Thêm dữ liệu từ bệnh viện Việt Nam để tăng tính khái quát")
    print(f"      → Đặc biệt thiếu nhóm tuổi trẻ (<40) — cần bổ sung")

    print(f"\n{'='*60}")
