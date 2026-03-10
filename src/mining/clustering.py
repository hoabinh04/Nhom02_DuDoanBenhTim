"""
src/mining/clustering.py — Phân cụm K-Means / DBSCAN + Elbow + Silhouette.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from src import load_params, get_path


def run_clustering(X: pd.DataFrame, y: pd.Series = None, params: dict = None) -> dict:
    """
    Phân cụm K-Means (+ Elbow & Silhouette) và DBSCAN.
    Trả về dict kết quả.
    """
    if params is None:
        params = load_params()

    k_range = params["mining"]["clustering"]["n_clusters_range"]
    seed = params["seed"]
    fig_dir = get_path(params["paths"]["figures_dir"])

    print("\n--- PHÂN CỤM (Clustering) ---")
    results = {}

    # K-Means: tìm K tối ưu
    inertias, silhouettes, dbis = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        silhouettes.append(sil)
        dbis.append(dbi)
        print(f"  K-Means k={k}: Silhouette={sil:.4f}, DBI={dbi:.4f}")

    best_k = k_range[np.argmax(silhouettes)]
    print(f"  → K tối ưu: {best_k} (Silhouette={max(silhouettes):.4f}, DBI={dbis[np.argmax(silhouettes)]:.4f})")

    km_best = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    km_labels = km_best.fit_predict(X)
    results["kmeans_labels"] = km_labels
    results["kmeans_model"] = km_best
    results["best_k"] = best_k
    results["silhouette_scores"] = dict(zip(k_range, silhouettes))
    results["dbi_scores"] = dict(zip(k_range, dbis))

    # DBSCAN
    db = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X)
    n_cl = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    print(f"  DBSCAN: {n_cl} cụm, {n_noise} nhiễu")
    results["dbscan_labels"] = db_labels

    # --- Hồ sơ cụm nguy cơ (Cluster Risk Profiling) ---
    if y is not None:
        print(f"\n  HỒ SƠ CỤM NGUY CƠ (K-Means k={best_k}):")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        profile = X_df.copy()
        profile["cluster"] = km_labels
        profile["target"] = y.values if hasattr(y, "values") else y
        for c in range(best_k):
            mask = profile["cluster"] == c
            sub = profile[mask]
            n_total = len(sub)
            n_disease = (sub["target"] == 1).sum()
            risk_pct = n_disease / n_total * 100 if n_total > 0 else 0
            print(f"\n    Cụm {c}: {n_total} mẫu, {n_disease} có bệnh ({risk_pct:.1f}%)")
            means = sub.drop(columns=["cluster", "target"]).mean()
            top_feats = means.abs().nlargest(5)
            for feat, val in top_feats.items():
                print(f"      {feat}: mean = {val:.3f}")
            if risk_pct >= 50:
                print(f"      → ⚠ CỤM NGUY CƠ CAO")
            else:
                print(f"      → Cụm nguy cơ thấp")
        results["cluster_profile"] = profile

    # --- Biểu đồ ---
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(k_range, inertias, "bo-")
    axes[0, 0].set_title("Elbow Method", fontweight="bold")
    axes[0, 0].set_xlabel("K"); axes[0, 0].set_ylabel("Inertia")

    axes[0, 1].plot(k_range, silhouettes, "rs-")
    axes[0, 1].set_title("Silhouette Score", fontweight="bold")
    axes[0, 1].set_xlabel("K"); axes[0, 1].set_ylabel("Score")
    axes[0, 1].axvline(best_k, color="green", ls="--", label=f"Best K={best_k}")
    axes[0, 1].legend()

    axes[0, 2].plot(k_range, dbis, "g^-")
    axes[0, 2].set_title("Davies-Bouldin Index (thấp hơn = tốt hơn)", fontweight="bold")
    axes[0, 2].set_xlabel("K"); axes[0, 2].set_ylabel("DBI")
    axes[0, 2].axvline(best_k, color="green", ls="--", label=f"Best K={best_k}")
    axes[0, 2].legend()

    sc = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap="Set2", s=15, alpha=0.7)
    axes[1, 0].set_title(f"K-Means k={best_k} (PCA 2D)", fontweight="bold")
    plt.colorbar(sc, ax=axes[1, 0], label="Cụm")

    if y is not None:
        sc2 = axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=y.values, cmap="RdYlGn_r", s=15, alpha=0.7)
        axes[1, 1].set_title("Target thật (PCA 2D)", fontweight="bold")
        plt.colorbar(sc2, ax=axes[1, 1], label="Target")
    else:
        sc2 = axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=db_labels, cmap="Set2", s=15, alpha=0.7)
        axes[1, 1].set_title("DBSCAN (PCA 2D)", fontweight="bold")
        plt.colorbar(sc2, ax=axes[1, 1], label="Cụm")

    sc3 = axes[1, 2].scatter(X_2d[:, 0], X_2d[:, 1], c=db_labels, cmap="Set2", s=15, alpha=0.7)
    axes[1, 2].set_title(f"DBSCAN ({n_cl} cụm, {n_noise} nhiễu)", fontweight="bold")
    plt.colorbar(sc3, ax=axes[1, 2], label="Cụm")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "clustering_results.png"), bbox_inches="tight")
    plt.show()

    return results
