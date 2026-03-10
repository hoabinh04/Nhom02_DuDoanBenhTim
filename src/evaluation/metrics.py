"""
src/evaluation/metrics.py — Tính metric đánh giá mô hình.

Metric ưu tiên cho bài toán y tế: PR-AUC, F1, Recall
+ Phân tích lỗi FN / FP chi tiết.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def compute_metrics(model, X_test, y_test, name: str) -> tuple:
    """
    Tính tất cả metric cho một mô hình.
    Trả về (dict_metrics, y_pred, y_proba).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    m = {
        "Mô hình": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "PR-AUC": round(average_precision_score(y_test, y_proba), 4) if y_proba is not None else None,
    }
    return m, y_pred, y_proba


def error_analysis(preds: dict, X_test, y_test) -> pd.DataFrame:
    """
    Phân tích lỗi FN / FP cho từng mô hình.
    + Profile đặc trưng của nhóm FN (bệnh nhân bị bỏ sót).
    Trả về DataFrame tổng hợp TP / TN / FP / FN + tỷ lệ.
    """
    print(f"\n{'='*60}")
    print("PHÂN TÍCH LỖI (FN / FP)")
    print(f"{'='*60}")

    rows = []
    y_arr = np.array(y_test)
    X_arr = np.array(X_test)
    feat_names = X_test.columns.tolist() if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_arr.shape[1])]

    best_model = None
    best_f1 = -1

    for name, p in preds.items():
        y_pred = p["y_pred"]
        cm = confusion_matrix(y_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()
        n = len(y_arr)
        rows.append({
            "Mô hình": name,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "FP Rate (%)": round(fp / n * 100, 2),
            "FN Rate (%)": round(fn / n * 100, 2),
        })

        f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

        print(f"\n  {name}:")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"    FP Rate={fp/n*100:.1f}%  FN Rate={fn/n*100:.1f}%")
        if fn > 0:
            print(f"    ⚠ {fn} bệnh nhân bị bỏ sót (False Negative — nguy hiểm trong y tế)")

    df_err = pd.DataFrame(rows)
    print(f"\n{'─'*50}")
    print(df_err.to_string(index=False))

    # Profile nhóm FN của mô hình tốt nhất
    if best_model is not None:
        y_pred_best = preds[best_model]["y_pred"]
        fn_mask = (y_arr == 1) & (y_pred_best == 0)
        tp_mask = (y_arr == 1) & (y_pred_best == 1)

        if fn_mask.sum() > 0 and tp_mask.sum() > 0:
            print(f"\n{'─'*50}")
            print(f"PROFILE NHÓM BỊ BỎ SÓT (FN) — {best_model}")
            print(f"{'─'*50}")
            print(f"  Số FN: {fn_mask.sum()}, Số TP: {tp_mask.sum()}")

            fn_data = pd.DataFrame(X_arr[fn_mask], columns=feat_names)
            tp_data = pd.DataFrame(X_arr[tp_mask], columns=feat_names)

            print(f"\n  So sánh mean(FN) vs mean(TP) — đặc trưng khác biệt lớn nhất:")
            diff = (fn_data.mean() - tp_data.mean()).abs().sort_values(ascending=False)
            for feat in diff.head(5).index:
                fn_mean = fn_data[feat].mean()
                tp_mean = tp_data[feat].mean()
                print(f"    {feat}: FN mean={fn_mean:.3f}, TP mean={tp_mean:.3f}, |Δ|={abs(fn_mean-tp_mean):.3f}")

            print(f"\n  Dạng sai phổ biến:")
            print(f"    → Nhóm FN có profile 'giống không bệnh' ở một số đặc trưng")
            print(f"    → Đây là nhóm bệnh nhân khó phân loại (borderline cases)")
            print(f"    → Khuyến nghị: tăng Recall bằng cách hạ ngưỡng quyết định hoặc dùng ensemble")

    return df_err


def evaluate_all(model_results: dict, X_test, y_test) -> tuple:
    """
    Đánh giá tất cả mô hình trên tập test.
    Trả về (comparison_df, predictions_dict).
    """
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)

    all_m = []
    preds = {}

    for name, res in model_results.items():
        model = res["model"]
        m, y_pred, y_proba = compute_metrics(model, X_test, y_test, name)
        all_m.append(m)
        preds[name] = {"y_pred": y_pred, "y_proba": y_proba}

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Có bệnh"]))

    comp = pd.DataFrame(all_m).sort_values("F1-Score", ascending=False)

    print(f"\n{'='*60}")
    print("BẢNG SO SÁNH")
    print(f"{'='*60}")
    print(comp.to_string(index=False))

    # Phân tích lỗi
    df_err = error_analysis(preds, X_test, y_test)

    return comp, preds
