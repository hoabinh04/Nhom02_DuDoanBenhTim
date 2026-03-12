"""
src/mining/association.py — Khai phá luật kết hợp (Apriori).
"""

import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src import load_params, get_path


def discretize_for_rules(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Rời rạc hóa biến số thành nhãn để dùng Apriori."""
    d = pd.DataFrame()
    target = params["data"]["target_col"]

    if "age" in df.columns:
        d["age_group"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 100],
                                labels=["age_young", "age_middle", "age_senior", "age_elderly"])
    if "trestbps" in df.columns:
        d["bp_level"] = pd.cut(df["trestbps"], bins=[0, 120, 140, 300],
                               labels=["bp_normal", "bp_elevated", "bp_high"])
    if "chol" in df.columns:
        d["chol_level"] = pd.cut(df["chol"], bins=[0, 200, 240, 700],
                                 labels=["chol_normal", "chol_borderline", "chol_high"])
    if "thalch" in df.columns:
        d["hr_level"] = pd.cut(df["thalch"], bins=[0, 120, 150, 220],
                               labels=["hr_low", "hr_normal", "hr_high"])

    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
        if col in df.columns:
            d[col] = df[col].astype(str)

    if target in df.columns:
        d["target"] = df[target].apply(lambda x: "has_disease" if x > 0 else "no_disease")

    return d


def mine_rules(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Tìm luật kết hợp bằng Apriori.
    Trả về DataFrame các luật.
    """
    if params is None:
        params = load_params()

    min_sup = params["mining"]["association"]["min_support"]
    min_conf = params["mining"]["association"]["min_confidence"]
    save_dir = get_path(params["paths"]["tables_dir"])

    print("\n--- LUẬT KẾT HỢP (Apriori) ---")

    df_disc = discretize_for_rules(df, params)

    # Chuyển thành giao dịch
    transactions = []
    for _, row in df_disc.iterrows():
        items = [str(v) for v in row.values if pd.notna(v) and str(v) != "nan"]
        transactions.append(items)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(df_onehot, min_support=min_sup, use_colnames=True)
    print(f"  Tập phổ biến: {len(frequent)} (min_support={min_sup})")

    if len(frequent) == 0:
        print("  ⚠ Không tìm được. Hãy thử giảm min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent, num_itemsets=len(frequent), metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("lift", ascending=False)
    print(f"  Luật kết hợp: {len(rules)} (min_confidence={min_conf})")

    # Lọc luật liên quan bệnh tim
    disease_rules = rules[
        rules["consequents"].apply(lambda x: "has_disease" in str(x) or "no_disease" in str(x))
    ]
    print(f"  Liên quan bệnh tim: {len(disease_rules)}")

    # Lưu
    if len(rules) > 0:
        save = rules.copy()
        save["antecedents"] = save["antecedents"].apply(lambda x: ", ".join(list(x)))
        save["consequents"] = save["consequents"].apply(lambda x: ", ".join(list(x)))
        save.to_csv(os.path.join(save_dir, "association_rules.csv"), index=False)

    # In top luật
    if len(disease_rules) > 0:
        print("\n  Top 10 luật (theo Lift):")
        for _, row in disease_rules.head(10).iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            print(f"    {ant} → {con}  "
                  f"[sup={row['support']:.3f}, conf={row['confidence']:.3f}, lift={row['lift']:.3f}]")

        # Diễn giải / Insight
        print(f"\n  DIỄN GIẢI TỔ HỢP TRIỆU CHỨNG:")
        top3 = disease_rules.head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            print(f"\n    [{i}] Nếu bệnh nhân có: {ant}")
            print(f"        → Kết luận: {con}")
            print(f"        Độ tin cậy: {row['confidence']*100:.1f}%, "
                  f"Lift: {row['lift']:.2f}x so với ngẫu nhiên")
            if row["lift"] > 2:
                print(f"        ⚠ Tổ hợp có mối liên hệ rất mạnh (lift > 2)")

        # Gợi ý combo/cross-sell kiểu y tế
        has_disease = disease_rules[disease_rules["consequents"].apply(lambda x: "has_disease" in str(x))]
        if len(has_disease) > 0:
            print(f"\n  GỢI Ý SÀNG LỌC:")
            print(f"    Các tổ hợp triệu chứng sau nên được ưu tiên kiểm tra:")
            for _, row in has_disease.head(5).iterrows():
                ant = ", ".join(list(row["antecedents"]))
                print(f"    • {ant}  (conf={row['confidence']:.1%}, lift={row['lift']:.2f})")

    return rules
