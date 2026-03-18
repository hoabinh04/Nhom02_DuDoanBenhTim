"""
src/mining/anomaly.py — Phát hiện ngoại lệ (Isolation Forest).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from src import load_params, get_path


def detect_anomalies(X: pd.DataFrame, params: dict = None) -> np.ndarray:
    """
    Phát hiện ngoại lệ bằng Isolation Forest.
    Trả về array: 1 = bình thường, -1 = ngoại lệ.
    """
    if params is None:
        params = load_params()

    contamination = params["mining"]["anomaly"]["contamination"]
    seed = params["seed"]
    fig_dir = get_path(params["paths"]["figures_dir"])

    print("\n--- PHÁT HIỆN NGOẠI LỆ (Isolation Forest) ---")

    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    labels = iso.fit_predict(X)

    n_anomaly = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    print(f"  Bình thường: {n_normal}, Ngoại lệ: {n_anomaly} ({n_anomaly/len(X)*100:.1f}%)")

    # Biểu đồ
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if l == 1 else "#e74c3c" for l in labels]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=15, alpha=0.7)
    ax.set_title("Isolation Forest (PCA 2D)", fontweight="bold", fontsize=13)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", label=f"Bình thường ({n_normal})"),
        Patch(facecolor="#e74c3c", label=f"Ngoại lệ ({n_anomaly})"),
    ], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "anomaly_detection.png"), bbox_inches="tight")
    plt.show()

    return labels
