"""
Streamlit Dashboard — BTL Data Mining: Dự đoán Bệnh Tim (UCI Heart Disease)
Chạy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FIGURES = BASE / "outputs" / "figures"
TABLES = BASE / "outputs" / "tables"
MODELS = BASE / "outputs" / "models"
RAW_DATA = BASE / "data" / "raw" / "heart_disease_uci.csv"
PROCESSED = BASE / "data" / "processed"

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán Bệnh Tim — Dashboard",
    page_icon="❤️",
    layout="wide",
)

# ── Sidebar navigation ──────────────────────────────────────────────────
st.sidebar.title("📊 BTL Data Mining")
st.sidebar.markdown("**Đề tài 2:** Dự đoán Bệnh Tim")
st.sidebar.markdown("---")

pages = [
    "🏠 Tổng quan",
    "📈 Khám phá dữ liệu (EDA)",
    "⛏️ Khai phá dữ liệu",
    "🤖 Mô hình phân loại",
    "🔄 Học bán giám sát",
    "📉 Mô hình hồi quy",
    "🔮 Dự đoán trực tuyến",
]
page = st.sidebar.radio("Chọn trang", pages)


# ── Helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def show_image(name, caption=None):
    img_path = FIGURES / name
    if img_path.exists():
        st.image(str(img_path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Không tìm thấy: {name}")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 1: TỔNG QUAN
# ═════════════════════════════════════════════════════════════════════════
if page == pages[0]:
    st.title("❤️ Dự đoán Bệnh Tim — Tổng quan dự án")
    st.markdown("""
    Dự án sử dụng tập dữ liệu **UCI Heart Disease** để xây dựng các mô hình
    Machine Learning dự đoán nguy cơ bệnh tim. Pipeline bao gồm:
    - Tiền xử lý & làm sạch dữ liệu
    - Kỹ thuật đặc trưng (Feature Engineering)
    - Khai phá luật kết hợp (Apriori), phân cụm (K-Means, DBSCAN), phát hiện bất thường
    - Huấn luyện 7 mô hình phân loại có giám sát
    - Học bán giám sát (Self-Training, Label Spreading)
    - Hồi quy dự đoán mức độ bệnh
    """)

    # Dataset overview
    st.header("📋 Thông tin tập dữ liệu")
    if RAW_DATA.exists():
        df_raw = load_csv(RAW_DATA)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số mẫu", f"{len(df_raw):,}")
        c2.metric("Số đặc trưng", f"{df_raw.shape[1] - 1}")
        c3.metric("Bệnh tim (num > 0)", f"{(df_raw['num'] > 0).sum():,}")
        c4.metric("Không bệnh (num = 0)", f"{(df_raw['num'] == 0).sum():,}")

        with st.expander("👀 Xem dữ liệu thô (5 dòng đầu)"):
            st.dataframe(df_raw.head(), use_container_width=True)

        with st.expander("📊 Thống kê mô tả"):
            st.dataframe(df_raw.describe(), use_container_width=True)

        # Missing values
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            with st.expander("⚠️ Giá trị thiếu"):
                st.dataframe(
                    pd.DataFrame({"Cột": missing.index, "Số lượng thiếu": missing.values,
                                  "Tỷ lệ (%)": (missing.values / len(df_raw) * 100).round(2)}),
                    use_container_width=True,
                )
        else:
            st.success("Dữ liệu không có giá trị thiếu.")
    else:
        st.error("Không tìm thấy file dữ liệu. Hãy chạy pipeline trước.")

    # Quick results summary
    st.header("🏆 Tóm tắt kết quả")
    comp_path = TABLES / "model_comparison.csv"
    if comp_path.exists():
        df_comp = load_csv(comp_path)
        best_row = df_comp.loc[df_comp["F1-Score"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mô hình tốt nhất (F1)", best_row["Mô hình"], f"F1 = {best_row['F1-Score']:.4f}")
        c2.metric("AUC-ROC cao nhất", df_comp.loc[df_comp["AUC-ROC"].idxmax(), "Mô hình"],
                   f"{df_comp['AUC-ROC'].max():.4f}")
        c3.metric("PR-AUC cao nhất", df_comp.loc[df_comp["PR-AUC"].idxmax(), "Mô hình"],
                   f"{df_comp['PR-AUC'].max():.4f}")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 2: KHÁM PHÁ DỮ LIỆU (EDA)
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[1]:
    st.title("📈 Khám phá dữ liệu (EDA)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Phân phối biến mục tiêu",
        "Phân phối biến số",
        "Phân phối biến phân loại",
        "Ma trận tương quan",
        "Biến số theo nhóm bệnh",
    ])

    with tab1:
        st.subheader("Phân phối biến mục tiêu (num)")
        show_image("target_distribution.png", "Phân phối biến mục tiêu")
        st.markdown("""
        **Nhận xét:** Tỷ lệ giữa hai lớp (bệnh/không bệnh) tương đối cân bằng,
        tuy nhiên vẫn cần áp dụng `class_weight='balanced'` để đảm bảo mô hình
        không thiên lệch.
        """)

    with tab2:
        st.subheader("Phân phối các biến số liên tục")
        show_image("numerical_distributions.png", "Phân phối biến số")

    with tab3:
        st.subheader("Phân phối các biến phân loại")
        show_image("categorical_distributions.png", "Phân phối biến phân loại")

    with tab4:
        st.subheader("Ma trận tương quan")
        show_image("correlation_matrix.png", "Ma trận tương quan Pearson")
        st.markdown("""
        **Nhận xét:** Các đặc trưng `cp`, `thalch`, `exang`, `oldpeak`, `ca`, `thal`
        có tương quan đáng kể với biến mục tiêu.
        """)

    with tab5:
        st.subheader("So sánh biến số theo nhóm bệnh")
        show_image("numerical_by_target.png", "Phân phối biến số theo nhóm bệnh/không bệnh")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 3: KHAI PHÁ DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[2]:
    st.title("⛏️ Khai phá dữ liệu")

    tab1, tab2, tab3 = st.tabs(["Luật kết hợp (Apriori)", "Phân cụm", "Phát hiện bất thường"])

    with tab1:
        st.subheader("Luật kết hợp — Thuật toán Apriori")
        rules_path = TABLES / "association_rules.csv"
        if rules_path.exists():
            df_rules = load_csv(rules_path)

            # Filters
            c1, c2, c3 = st.columns(3)
            min_conf = c1.slider("Confidence tối thiểu", 0.0, 1.0, 0.5, 0.05)
            min_lift = c2.slider("Lift tối thiểu", 0.0, 5.0, 1.0, 0.1)
            min_sup = c3.slider("Support tối thiểu", 0.0, 0.5, 0.0, 0.01)

            filtered = df_rules[
                (df_rules["confidence"] >= min_conf) &
                (df_rules["lift"] >= min_lift) &
                (df_rules["support"] >= min_sup)
            ]

            st.info(f"Hiển thị **{len(filtered)}** / {len(df_rules)} luật")

            display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
            st.dataframe(
                filtered[display_cols].sort_values("lift", ascending=False).reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "support": st.column_config.NumberColumn(format="%.4f"),
                    "confidence": st.column_config.NumberColumn(format="%.4f"),
                    "lift": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        else:
            st.warning("Chưa có kết quả luật kết hợp. Hãy chạy pipeline.")

    with tab2:
        st.subheader("Phân cụm — K-Means & DBSCAN")
        show_image("clustering_results.png", "Kết quả phân cụm")
        st.markdown("""
        **Nhận xét:** Kết quả phân cụm giúp phát hiện các nhóm bệnh nhân có đặc điểm
        lâm sàng tương tự. Cluster risk profiling cho thấy mỗi cụm có tỷ lệ bệnh tim khác nhau.
        """)

    with tab3:
        st.subheader("Phát hiện bất thường — Isolation Forest")
        show_image("anomaly_detection.png", "Phát hiện bất thường bằng Isolation Forest + PCA")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 4: MÔ HÌNH PHÂN LOẠI
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[3]:
    st.title("🤖 Mô hình phân loại có giám sát")

    tab1, tab2, tab3, tab4 = st.tabs([
        "So sánh mô hình", "ROC & PR Curves", "Confusion Matrices", "Chi tiết huấn luyện"
    ])

    with tab1:
        st.subheader("Bảng so sánh các mô hình")
        comp_path = TABLES / "model_comparison.csv"
        if comp_path.exists():
            df_comp = load_csv(comp_path)

            # Highlight best values
            st.dataframe(
                df_comp.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC"],
                    color="#90EE90",
                ),
                use_container_width=True,
            )

            # Bar chart
            st.subheader("Biểu đồ so sánh")
            show_image("model_comparison_bar.png", "So sánh các chỉ số đánh giá")

            # Select metric to sort
            metric = st.selectbox("Sắp xếp theo chỉ số:", 
                                  ["F1-Score", "Accuracy", "Precision", "Recall", "AUC-ROC", "PR-AUC"])
            st.bar_chart(
                df_comp.set_index("Mô hình")[[metric]].sort_values(metric, ascending=True),
            )
        else:
            st.warning("Chưa có kết quả. Hãy chạy pipeline.")

    with tab2:
        st.subheader("Đường cong ROC")
        show_image("roc_curves.png", "ROC Curves cho tất cả mô hình")

        st.subheader("Đường cong Precision-Recall")
        show_image("pr_curves.png", "Precision-Recall Curves")
        st.markdown("""
        **Nhận xét:** PR-AUC quan trọng hơn khi dữ liệu mất cân bằng. RandomForest đạt
        PR-AUC cao nhất, cho thấy khả năng phát hiện bệnh tim ổn định ở nhiều ngưỡng.
        """)

    with tab3:
        st.subheader("Ma trận nhầm lẫn (Confusion Matrices)")
        show_image("confusion_matrices.png", "Confusion Matrices cho tất cả mô hình")
        st.markdown("""
        **Nhận xét:** Trong y tế, **False Negative** (bỏ sót bệnh nhân) nghiêm trọng hơn
        False Positive. Mô hình SVM có Recall cao nhất (0.892), bỏ sót ít bệnh nhân nhất.
        """)

    with tab4:
        st.subheader("Chi tiết huấn luyện & Siêu tham số tốt nhất")
        train_path = TABLES / "model_training_results.csv"
        if train_path.exists():
            df_train = load_csv(train_path)
            st.dataframe(df_train, use_container_width=True)

            st.markdown("### Siêu tham số tốt nhất")
            for _, row in df_train.iterrows():
                with st.expander(f"**{row['Mô hình']}** — CV F1: {row['CV F1 (mean)']:.4f}"):
                    st.code(row["Best Params"])
                    st.write(f"- CV F1 std: {row['CV F1 (std)']:.4f}")
                    st.write(f"- Thời gian huấn luyện: {row['Thời gian (s)']:.2f}s")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 5: HỌC BÁN GIÁM SÁT
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[4]:
    st.title("🔄 Học bán giám sát (Semi-supervised Learning)")

    semi_path = TABLES / "semi_supervised_results.csv"
    if semi_path.exists():
        df_semi = load_csv(semi_path)

        # Summary metrics
        st.subheader("📊 Bảng kết quả tổng hợp")
        st.dataframe(
            df_semi.style.highlight_max(subset=["Accuracy", "F1", "PR-AUC"], color="#90EE90"),
            use_container_width=True,
        )

        # Learning curve
        st.subheader("📈 Learning Curve")
        show_image("semi_supervised_learning_curve.png", "Semi-supervised Learning Curve")

        # Pivot tables
        st.subheader("🔍 Phân tích chi tiết")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**F1-Score theo tỷ lệ nhãn & phương pháp**")
            pivot_f1 = df_semi.pivot_table(
                index="Tỷ lệ nhãn (%)", columns="Phương pháp", values="F1"
            )
            st.dataframe(pivot_f1.style.highlight_max(axis=1, color="#90EE90"), use_container_width=True)

        with c2:
            st.markdown("**PR-AUC theo tỷ lệ nhãn & phương pháp**")
            pivot_pr = df_semi.pivot_table(
                index="Tỷ lệ nhãn (%)", columns="Phương pháp", values="PR-AUC"
            )
            st.dataframe(pivot_pr.style.highlight_max(axis=1, color="#90EE90"), use_container_width=True)

        # Pseudo-label accuracy
        self_train = df_semi[df_semi["Phương pháp"] == "Self-Training"]
        if not self_train.empty and "Pseudo-label Acc (%)" in self_train.columns:
            st.subheader("🏷️ Độ chính xác Pseudo-label (Self-Training)")
            st.bar_chart(
                self_train.set_index("Tỷ lệ nhãn (%)")["Pseudo-label Acc (%)"],
            )

        st.markdown("""
        **Nhận xét:**
        - **Supervised-only** luôn vượt trội hơn semi-supervised ở tất cả tỷ lệ nhãn
        - **Self-Training** cho kết quả khá tốt khi tỷ lệ nhãn ≥ 20%
        - **Label Spreading** có xu hướng kém hơn do giả định cấu trúc manifold không khớp dữ liệu y tế
        - Pseudo-label accuracy dao động 65–73%, cho thấy noise propagation đáng kể
        """)
    else:
        st.warning("Chưa có kết quả. Hãy chạy pipeline.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 6: MÔ HÌNH HỒI QUY
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[5]:
    st.title("📉 Mô hình hồi quy — Dự đoán mức độ bệnh")

    reg_path = TABLES / "regression_results.csv"
    if reg_path.exists():
        df_reg = load_csv(reg_path)

        st.subheader("📊 Bảng kết quả hồi quy")
        st.dataframe(
            df_reg.style.highlight_min(
                subset=["CV MAE (mean)", "CV RMSE (mean)"], color="#90EE90"
            ),
            use_container_width=True,
        )

        # Overfitting analysis
        st.subheader("⚠️ Phân tích Overfitting")
        df_reg["MAE Gap"] = (df_reg["CV MAE (mean)"] - df_reg["Train MAE"]).round(4)
        df_reg["RMSE Gap"] = (df_reg["CV RMSE (mean)"] - df_reg["Train RMSE"]).round(4)
        st.dataframe(
            df_reg[["Mô hình", "Train MAE", "CV MAE (mean)", "MAE Gap", "Train RMSE", "CV RMSE (mean)", "RMSE Gap"]],
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            show_image("regression_actual_vs_pred.png", "Actual vs Predicted")
        with c2:
            show_image("regression_residuals.png", "Phân phối Residuals")
        with c3:
            show_image("regression_feature_importance.png", "Feature Importance")

        st.markdown("""
        **Nhận xét:**
        - **Ridge & LinearRegression** cho kết quả gần như giống nhau (MAE ≈ 13.69), generalize tốt
        - **XGBRegressor** overfit nặng: Train MAE = 5.44 vs CV MAE = 15.56
        - Bài toán hồi quy khó hơn phân loại vì target gốc (0–4) mang tính thứ tự hơn là liên tục
        """)
    else:
        st.warning("Chưa có kết quả. Hãy chạy pipeline.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 7: DỰ ĐOÁN TRỰC TUYẾN
# ═════════════════════════════════════════════════════════════════════════
elif page == pages[6]:
    st.title("🔮 Dự đoán bệnh tim trực tuyến")
    st.markdown("Nhập thông tin bệnh nhân bên dưới để dự đoán nguy cơ bệnh tim.")

    # Load best model (GradientBoosting has best F1)
    model_path = MODELS / "GradientBoosting_best.pkl"
    if not model_path.exists():
        st.error("Chưa có mô hình. Hãy chạy pipeline trước (`python scripts/run_pipeline.py`).")
    else:
        # Load processed data to get feature names
        X_train_path = PROCESSED / "X_train.csv"
        if X_train_path.exists():
            X_train = load_csv(X_train_path)
            feature_names = list(X_train.columns)
        else:
            st.error("Không tìm thấy dữ liệu đã xử lý.")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        st.info("**Mô hình:** GradientBoosting (F1 = 0.8683)")

        # Input form
        with st.form("prediction_form"):
            st.subheader("Nhập thông tin bệnh nhân")

            c1, c2, c3 = st.columns(3)

            with c1:
                age = st.number_input("Tuổi (age)", min_value=20, max_value=100, value=55)
                trestbps = st.number_input("Huyết áp lúc nghỉ (trestbps)", 80, 220, 130)
                chol = st.number_input("Cholesterol (chol)", 100, 600, 250)
                fbs = st.selectbox("Đường huyết > 120 mg/dl (fbs)", [0, 1], format_func=lambda x: "Có" if x else "Không")

            with c2:
                sex = st.selectbox("Giới tính (sex)", [0, 1], format_func=lambda x: "Nam" if x else "Nữ")
                restecg = st.selectbox("Kết quả ECG (restecg)", [0, 1, 2])
                thalch = st.number_input("Nhịp tim tối đa (thalch)", 60, 220, 150)
                exang = st.selectbox("Đau thắt ngực khi vận động (exang)", [0, 1], format_func=lambda x: "Có" if x else "Không")

            with c3:
                oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
                slope = st.selectbox("Slope of ST segment", [0, 1, 2])
                ca = st.selectbox("Số mạch (ca)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia (thal)", [0, 1, 2])
                cp = st.selectbox("Kiểu đau ngực (cp)", [0, 1, 2, 3])

            submitted = st.form_submit_button("🔍 Dự đoán", use_container_width=True)

        if submitted:
            # Build input matching the trained features
            raw_input = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalch": thalch,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }

            # Create DataFrame with all features set to 0, then fill known values
            input_df = pd.DataFrame([{f: 0 for f in feature_names}])
            for col in raw_input:
                if col in input_df.columns:
                    input_df[col] = raw_input[col]

            try:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]

                st.markdown("---")
                st.subheader("📋 Kết quả dự đoán")

                if prediction == 1:
                    st.error(f"⚠️ **CÓ NGUY CƠ BỆNH TIM** — Xác suất: {proba[1]:.1%}")
                else:
                    st.success(f"✅ **KHÔNG CÓ NGUY CƠ BỆNH TIM** — Xác suất bình thường: {proba[0]:.1%}")

                c1, c2 = st.columns(2)
                c1.metric("Xác suất không bệnh", f"{proba[0]:.1%}")
                c2.metric("Xác suất bệnh tim", f"{proba[1]:.1%}")

                st.caption("⚠️ Kết quả chỉ mang tính tham khảo. Hãy tham vấn bác sĩ chuyên khoa.")
            except Exception as e:
                st.error(f"Lỗi dự đoán: {e}")


# ── Footer ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("BTL Data Mining — KHMT")
st.sidebar.caption("UCI Heart Disease Dataset")
