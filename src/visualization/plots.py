"""
src/visualization/plots.py  Hàm vẽ dùng chung cho toàn bộ dự án.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
)

from src import get_path

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")


def plot_target_distribution(df, target_col, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df[target_col].value_counts().sort_index()
    bars = axes[0].bar(counts.index.astype(str), counts.values, color=sns.color_palette("Set2"))
    axes[0].set_title("Phân bố Target gốc (0-4)", fontweight="bold")
    axes[0].set_xlabel("Mức độ")
    axes[0].set_ylabel("Số lượng")
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center")

    binary = (df[target_col] > 0).astype(int)
    c2 = binary.value_counts().sort_index()
    labels = ["Không bệnh (0)", "Có bệnh (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars2 = axes[1].bar(labels, c2.values, color=colors)
    axes[1].set_title("Target nhị phân", fontweight="bold")
    axes[1].set_ylabel("Số lượng")
    for b, v in zip(bars2, c2.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 5, f"{v} ({v/len(df)*100:.1f}%)", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"), bbox_inches="tight")
    plt.show()


def plot_numerical(df, num_cols, fig_dir):
    n = len(num_cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = np.array([axes])
    for i, col in enumerate(num_cols):
        axes[i, 0].hist(df[col].dropna(), bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        axes[i, 0].set_title(f"{col}", fontweight="bold")
        axes[i, 0].axvline(df[col].mean(), color="red", ls="--", label=f"Mean={df[col].mean():.1f}")
        axes[i, 0].legend(fontsize=9)
        axes[i, 1].boxplot(
            df[col].dropna(),
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#3498db", alpha=0.6),
        )
        axes[i, 1].set_title(f"Boxplot: {col}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "numerical_distributions.png"), bbox_inches="tight")
    plt.show()


def plot_categorical(df, cat_cols, target_col, fig_dir):
    n = len(cat_cols)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()
    t_label = (df[target_col] > 0).astype(int).map({0: "Không bệnh", 1: "Có bệnh"})
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], t_label)
        ct.plot(kind="bar", ax=axes[i], color=["#2ecc71", "#e74c3c"], edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].legend(title="Target", fontsize=8)
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Biến phân loại theo Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "categorical_distributions.png"), bbox_inches="tight")
    plt.show()


def plot_correlation(df, num_cols, target_col, fig_dir):
    cols = num_cols + [target_col]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Ma trận tương quan (Pearson)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_matrix.png"), bbox_inches="tight")
    plt.show()


def plot_violin_by_target(df, num_cols, target_col, fig_dir):
    temp = df.copy()
    temp["target_label"] = temp[target_col].apply(lambda x: "Có bệnh" if x > 0 else "Không bệnh")
    n = min(len(num_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(num_cols[:n]):
        sns.violinplot(
            data=temp,
            x="target_label",
            y=col,
            ax=axes[i],
            hue="target_label",
            palette={"Không bệnh": "#2ecc71", "Có bệnh": "#e74c3c"},
            inner="box",
            legend=False,
        )
        axes[i].set_title(f"{col}", fontweight="bold")
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("So sánh biến số theo Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "numerical_by_target.png"), bbox_inches="tight")
    plt.show()


def plot_model_comparison(comp_df, fig_dir):
    cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC"]
    df_plot = comp_df.set_index("Mô hình")[[c for c in cols if c in comp_df.columns]].dropna(axis=1)
    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot.plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
    ax.set_title("So sánh hiệu suất mô hình", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison_bar.png"), bbox_inches="tight")
    plt.show()


def plot_roc_curves(model_results, preds, X_test, y_test, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))
    for (name, _), color in zip(model_results.items(), colors):
        yp = preds[name]["y_proba"]
        if yp is not None:
            fpr, tpr, _ = roc_curve(y_test, yp)
            auc = roc_auc_score(y_test, yp)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC - So sánh mô hình", fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curves.png"), bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(preds, y_test, fig_dir):
    f1s = {n: f1_score(y_test, p["y_pred"]) for n, p in preds.items()}
    top = sorted(f1s, key=f1s.get, reverse=True)[:4]
    fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4))
    if len(top) == 1:
        axes = [axes]
    for ax, name in zip(axes, top):
        cm = confusion_matrix(y_test, preds[name]["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Không bệnh", "Có bệnh"],
            yticklabels=["Không bệnh", "Có bệnh"],
        )
        ax.set_title(f"{name}\nF1={f1s[name]:.3f}", fontweight="bold")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
    plt.suptitle("Confusion Matrix - Top mô hình", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), bbox_inches="tight")
    plt.show()


def plot_pr_curves(model_results, preds, X_test, y_test, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))
    for (name, _), color in zip(model_results.items(), colors):
        yp = preds[name]["y_proba"]
        if yp is not None:
            prec, rec, _ = precision_recall_curve(y_test, yp)
            ap = average_precision_score(y_test, yp)
            ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
    baseline = np.mean(np.array(y_test) == 1)
    ax.axhline(baseline, color="gray", ls="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_title("Precision-Recall Curve - So sánh mô hình", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curves.png"), bbox_inches="tight")
    plt.show()


def run_all_eda_plots(df, params):
    target = params["data"]["target_col"]
    num_cols = [c for c in params["data"]["numerical_cols"] if c in df.columns]
    cat_cols = [c for c in params["data"]["categorical_cols"] if c in df.columns]
    fig_dir = get_path(params["paths"]["figures_dir"])

    plot_target_distribution(df, target, fig_dir)
    plot_numerical(df, num_cols, fig_dir)
    plot_categorical(df, cat_cols, target, fig_dir)
    plot_correlation(df, num_cols, target, fig_dir)
    plot_violin_by_target(df, num_cols, target, fig_dir)
    print(f"[VIZ] EDA plots đã lưu -> {fig_dir}")


def run_all_eval_plots(comp_df, model_results, preds, X_test, y_test, params):
    fig_dir = get_path(params["paths"]["figures_dir"])
    plot_model_comparison(comp_df, fig_dir)
    plot_roc_curves(model_results, preds, X_test, y_test, fig_dir)
    plot_pr_curves(model_results, preds, X_test, y_test, fig_dir)
    plot_confusion_matrices(preds, y_test, fig_dir)
    print(f"[VIZ] Eval plots đã lưu -> {fig_dir}")
