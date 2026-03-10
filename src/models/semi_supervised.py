"""
src/models/semi_supervised.py — Thực nghiệm bán giám sát:
  - Giữ p% nhãn (5, 10, 20)
  - Supervised-only vs Self-Training vs Label Spreading
  - Learning curve + Phân tích rủi ro pseudo-label
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

from src import load_params, get_path


def _create_partial_labels(y, ratio, seed=42):
    """Giữ ratio% nhãn, còn lại = -1 (stratified)."""
    rng = np.random.RandomState(seed)
    y_part = y.copy().values.astype(float)
    labeled = []
    for cls in np.unique(y_part):
        idx = np.where(y_part == cls)[0]
        n = max(int(len(idx) * ratio), 1)
        labeled.extend(rng.choice(idx, size=n, replace=False))
    mask = np.ones(len(y_part), dtype=bool)
    mask[labeled] = False
    y_part[mask] = -1
    print(f"  Tỷ lệ {ratio*100:.0f}%: có nhãn {(y_part != -1).sum()}/{len(y_part)}")
    return y_part


def run_semi_supervised(X_train, y_train, X_test, y_test, params: dict = None) -> pd.DataFrame:
    """Chạy thực nghiệm bán giám sát cho tất cả tỷ lệ nhãn."""
    if params is None:
        params = load_params()

    ratios = params["semi_supervised"]["label_ratios"]
    seed = params["seed"]
    threshold = params["semi_supervised"]["self_training_threshold"]
    gamma = params["semi_supervised"]["label_spreading_gamma"]
    alpha = params["semi_supervised"]["label_spreading_alpha"]
    fig_dir = get_path(params["paths"]["figures_dir"])
    tables_dir = get_path(params["paths"]["tables_dir"])

    print("\n" + "=" * 60)
    print("THỰC NGHIỆM BÁN GIÁM SÁT")
    print("=" * 60)

    all_results = []
    X_arr = X_train.values if hasattr(X_train, "values") else X_train

    for ratio in ratios:
        print(f"\n{'─'*50}")
        print(f"TỶ LỆ NHÃN: {ratio*100:.0f}%")
        print(f"{'─'*50}")

        y_part = _create_partial_labels(y_train, ratio, seed)
        labeled_mask = y_part != -1
        X_lab = X_arr[labeled_mask]
        y_lab = y_part[labeled_mask].astype(int)

        # (A) Supervised-only
        print(f"\n  [A] Supervised-only ({labeled_mask.sum()} mẫu):")
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
        t0 = time.time()
        rf.fit(X_lab, y_lab)
        t_sup = time.time() - t0
        pred_sup = rf.predict(X_test)
        proba_sup = rf.predict_proba(X_test)[:, 1]
        f1_sup = f1_score(y_test, pred_sup)
        acc_sup = accuracy_score(y_test, pred_sup)
        pr_auc_sup = average_precision_score(y_test, proba_sup)
        print(f"    Acc={acc_sup:.4f}, F1={f1_sup:.4f}, PR-AUC={pr_auc_sup:.4f}, Time={t_sup:.2f}s")
        all_results.append({"Tỷ lệ nhãn (%)": int(ratio*100), "Phương pháp": "Supervised-only",
                            "Accuracy": round(acc_sup, 4), "F1": round(f1_sup, 4),
                            "PR-AUC": round(pr_auc_sup, 4), "Thời gian (s)": round(t_sup, 2)})

        # (B) Self-Training
        print(f"\n  [B] Self-Training (pseudo-label, threshold={threshold}):")
        base = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
        st = SelfTrainingClassifier(estimator=base, threshold=threshold, max_iter=20)
        t0 = time.time()
        st.fit(X_arr, y_part)
        t_st = time.time() - t0
        pred_st = st.predict(X_test)
        proba_st = st.predict_proba(X_test)[:, 1]
        f1_st = f1_score(y_test, pred_st)
        acc_st = accuracy_score(y_test, pred_st)
        pr_auc_st = average_precision_score(y_test, proba_st)
        print(f"    Acc={acc_st:.4f}, F1={f1_st:.4f}, PR-AUC={pr_auc_st:.4f}, Time={t_st:.2f}s")

        # Phân tích pseudo-label sai
        pseudo_labels = st.transduction_
        unlabeled_mask = y_part == -1
        y_true_unlabeled = np.array(y_train)[unlabeled_mask]
        pseudo_unlabeled = pseudo_labels[unlabeled_mask]
        n_pseudo = len(y_true_unlabeled)
        n_correct = (y_true_unlabeled == pseudo_unlabeled).sum()
        n_wrong = n_pseudo - n_correct
        pseudo_acc = n_correct / n_pseudo * 100 if n_pseudo > 0 else 0
        print(f"    Pseudo-label: {n_pseudo} mẫu, đúng {n_correct} ({pseudo_acc:.1f}%), sai {n_wrong}")
        if n_wrong > 0:
            wrong_mask = y_true_unlabeled != pseudo_unlabeled
            fn_pseudo = ((y_true_unlabeled == 1) & (pseudo_unlabeled == 0)).sum()
            fp_pseudo = ((y_true_unlabeled == 0) & (pseudo_unlabeled == 1)).sum()
            print(f"    Pseudo-label sai: FN={fn_pseudo} (bỏ sót bệnh), FP={fp_pseudo} (báo nhầm)")

        all_results.append({"Tỷ lệ nhãn (%)": int(ratio*100), "Phương pháp": "Self-Training",
                            "Accuracy": round(acc_st, 4), "F1": round(f1_st, 4),
                            "PR-AUC": round(pr_auc_st, 4), "Thời gian (s)": round(t_st, 2),
                            "Pseudo-label Acc (%)": round(pseudo_acc, 1)})

        # (C) Label Spreading
        print(f"\n  [C] Label Spreading (kernel=rbf, gamma={gamma}):")
        ls = LabelSpreading(kernel="rbf", gamma=gamma, max_iter=50, alpha=alpha)
        t0 = time.time()
        ls.fit(X_arr, y_part)
        t_ls = time.time() - t0
        pred_ls = ls.predict(X_test)
        proba_ls = ls.predict_proba(X_test)[:, 1]
        f1_ls = f1_score(y_test, pred_ls)
        acc_ls = accuracy_score(y_test, pred_ls)
        pr_auc_ls = average_precision_score(y_test, proba_ls)
        print(f"    Acc={acc_ls:.4f}, F1={f1_ls:.4f}, PR-AUC={pr_auc_ls:.4f}, Time={t_ls:.2f}s")
        all_results.append({"Tỷ lệ nhãn (%)": int(ratio*100), "Phương pháp": "Label Spreading",
                            "Accuracy": round(acc_ls, 4), "F1": round(f1_ls, 4),
                            "PR-AUC": round(pr_auc_ls, 4), "Thời gian (s)": round(t_ls, 2)})

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(tables_dir, "semi_supervised_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("BẢNG TỔNG HỢP BÁN GIÁM SÁT")
    print(f"{'='*60}")
    print(df_res.to_string(index=False))

    _plot_learning_curve(df_res, fig_dir)
    _analyze_risk(df_res)

    return df_res


def _plot_learning_curve(df, fig_dir):
    """Learning curve: F1, Accuracy, PR-AUC theo % nhãn."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colors = {"Supervised-only": "#3498db", "Self-Training": "#e74c3c", "Label Spreading": "#2ecc71"}

    for metric, ax in zip(["F1", "Accuracy", "PR-AUC"], axes):
        for method in df["Phương pháp"].unique():
            sub = df[df["Phương pháp"] == method]
            ax.plot(sub["Tỷ lệ nhãn (%)"], sub[metric], "o-",
                    label=method, color=colors.get(method, "gray"), linewidth=2, markersize=8)
        ax.set_title(f"Learning Curve — {metric}", fontweight="bold")
        ax.set_xlabel("% nhãn"); ax.set_ylabel(metric)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xticks(df["Tỷ lệ nhãn (%)"].unique())

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "semi_supervised_learning_curve.png"), bbox_inches="tight")
    plt.show()


def _analyze_risk(df):
    """Phân tích rủi ro pseudo-label + phân tích nhóm khó."""
    print(f"\n{'─'*50}")
    print("PHÂN TÍCH RỦI RO PSEUDO-LABEL")
    print(f"{'─'*50}")

    for r in df["Tỷ lệ nhãn (%)"].unique():
        sub = df[df["Tỷ lệ nhãn (%)"] == r]
        f1_sup = sub[sub["Phương pháp"] == "Supervised-only"]["F1"].values[0]
        f1_st = sub[sub["Phương pháp"] == "Self-Training"]["F1"].values[0]
        f1_ls = sub[sub["Phương pháp"] == "Label Spreading"]["F1"].values[0]
        pr_sup = sub[sub["Phương pháp"] == "Supervised-only"]["PR-AUC"].values[0]
        pr_st = sub[sub["Phương pháp"] == "Self-Training"]["PR-AUC"].values[0]
        pr_ls = sub[sub["Phương pháp"] == "Label Spreading"]["PR-AUC"].values[0]

        print(f"\n  {r}% nhãn:")
        print(f"    Supervised:      F1={f1_sup:.4f}  PR-AUC={pr_sup:.4f}")
        print(f"    Self-Training:   F1={f1_st:.4f} (Δ={f1_st - f1_sup:+.4f})  "
              f"PR-AUC={pr_st:.4f} (Δ={pr_st - pr_sup:+.4f})"
              + (" ⚠ kém hơn!" if f1_st < f1_sup else " ✓"))
        print(f"    Label Spreading: F1={f1_ls:.4f} (Δ={f1_ls - f1_sup:+.4f})  "
              f"PR-AUC={pr_ls:.4f} (Δ={pr_ls - pr_sup:+.4f})"
              + (" ⚠ kém hơn!" if f1_ls < f1_sup else " ✓"))

    # Tổng hợp nhận xét
    print(f"\n{'─'*50}")
    print("NHẬN XÉT TỔNG HỢP")
    print(f"{'─'*50}")
    best_5 = df[df["Tỷ lệ nhãn (%)"] == 5].sort_values("F1", ascending=False).iloc[0]
    best_20 = df[df["Tỷ lệ nhãn (%)"] == 20].sort_values("F1", ascending=False).iloc[0]
    gain = best_20["F1"] - best_5["F1"]
    print(f"  Cải thiện F1 từ 5% → 20% nhãn: {gain:+.4f}")
    if gain > 0.05:
        print(f"  → Thu thêm nhãn mang lại lợi ích đáng kể")
    else:
        print(f"  → Semi-supervised đã khai thác tốt dữ liệu unlabeled")
    print(f"  → Nhóm khó (ít nhãn nhất 5%): phương pháp tốt nhất là {best_5['Phương pháp']} (F1={best_5['F1']:.4f})")
    print(f"  → Cần cẩn trọng pseudo-label khi nhãn < 10% (rủi ro noise propagation)")
