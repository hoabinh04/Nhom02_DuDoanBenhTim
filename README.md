# Bài Tập Lớn Data Mining — Dự đoán Bệnh Tim (UCI Heart Disease)

## 1. Nguồn dữ liệu

- **Tên**: UCI Heart Disease Dataset
- **Nguồn**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **File**: `data/raw/heart_disease_uci.csv`
- **Số mẫu**: 920 bản ghi từ 4 bệnh viện (Cleveland, Hungary, Switzerland, VA Long Beach)
- **Số cột**: 16 (bao gồm id và target)

## 2. Data Dictionary

| Cột | Kiểu | Mô tả |
|-----|-------|--------|
| `id` | int | Mã định danh bệnh nhân |
| `age` | int | Tuổi bệnh nhân |
| `sex` | cat | Giới tính (Male / Female) |
| `dataset` | cat | Bệnh viện thu thập |
| `cp` | cat | Loại đau ngực |
| `trestbps` | float | Huyết áp lúc nghỉ (mmHg) |
| `chol` | float | Cholesterol huyết thanh (mg/dl) |
| `fbs` | bool | Đường huyết lúc đói > 120 mg/dl |
| `restecg` | cat | Kết quả điện tâm đồ lúc nghỉ |
| `thalch` | float | Nhịp tim tối đa đạt được |
| `exang` | bool | Đau ngực khi gắng sức |
| `oldpeak` | float | Độ chênh ST khi gắng sức |
| `slope` | cat | Độ dốc đoạn ST đỉnh gắng sức |
| `ca` | float | Số mạch máu lớn nhuộm huỳnh quang (0–3) |
| `thal` | cat | Thalassemia |
| `num` | int | **TARGET** — 0 = không bệnh, 1–4 = có bệnh (nhị phân hóa → 0/1) |

## 3. Cấu trúc thư mục

```
BTL-DATAMINING/
├── configs/
│   └── params.yaml                        # Toàn bộ tham số (paths, models, mining, …)
├── data/
│   ├── raw/                               # Dữ liệu gốc (.csv)
│   └── processed/                         # Dữ liệu đã xử lý (tự động tạo)
├── notebooks/
│   ├── 01_eda.ipynb                       # Khám phá dữ liệu (EDA)
│   ├── 02_preprocess_feature.ipynb        # Tiền xử lý & Feature Engineering
│   ├── 03_mining_or_clustering.ipynb      # Khai phá: Apriori, Clustering, Anomaly
│   ├── 04_modeling.ipynb                  # Huấn luyện 7 mô hình phân lớp
│   ├── 04b_semi_supervised.ipynb          # Thực nghiệm bán giám sát
│   ├── 05_evaluation_report.ipynb         # Đánh giá & Báo cáo tổng hợp
│   └── 06_regression.ipynb                # Hồi quy chỉ số (huyết áp)
├── outputs/
│   ├── figures/                           # Biểu đồ (.png)
│   ├── models/                            # Mô hình đã train (.pkl)
│   └── tables/                            # Bảng kết quả (.csv)
├── scripts/
│   └── run_pipeline.py                    # Chạy toàn bộ pipeline từ CLI
├── src/
│   ├── __init__.py                        # load_params(), get_path()
│   ├── data/
│   │   ├── loader.py                      # Đọc & kiểm tra dữ liệu
│   │   └── cleaner.py                     # Tiền xử lý, mã hóa, chuẩn hóa, SMOTE
│   ├── features/
│   │   └── builder.py                     # Tạo & chọn đặc trưng
│   ├── mining/
│   │   ├── association.py                 # Luật kết hợp (Apriori)
│   │   ├── clustering.py                  # K-Means, DBSCAN
│   │   └── anomaly.py                     # Isolation Forest
│   ├── models/
│   │   ├── supervised.py                  # 7 mô hình + GridSearchCV
│   │   ├── semi_supervised.py             # Self-Training, Label Spreading
│   │   └── regression.py                  # Hồi quy: Linear, Ridge, XGBRegressor
│   ├── evaluation/
│   │   ├── metrics.py                     # Tính metric (Acc, P, R, F1, AUC, PR-AUC) + Error Analysis
│   │   └── report.py                      # Lưu bảng, in insight
│   └── visualization/
│       └── plots.py                       # Tất cả hàm vẽ (EDA + Eval)
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Quy trình thực hiện

| Bước | Mô tả | Notebook |
|------|--------|----------|
| 1 | **EDA** — Khám phá phân bố, tương quan, missing | `01_eda` |
| 2 | **Tiền xử lý** — Missing, encode, scale, SMOTE, feature engineering | `02_preprocess_feature` |
| 3 | **Khai phá** — Apriori, K-Means, DBSCAN, Isolation Forest | `03_mining_or_clustering` |
| 4 | **Mô hình** — LR, DT, RF, SVM, KNN, GB, XGBoost + GridSearchCV | `04_modeling` |
| 4b | **Bán giám sát** — Supervised-only vs Self-Training vs Label Spreading (5/10/20/30%) | `04b_semi_supervised` |
| 5 | **Đánh giá** — So sánh, ROC, PR-AUC, Confusion Matrix, Error Analysis, Insight | `05_evaluation_report` |
| 6 | **Hồi quy** — Linear/Ridge/XGBRegressor dự đoán huyết áp, kiểm tra outlier/leakage | `06_regression` |

## 5. Cách chạy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2a. Chạy toàn bộ pipeline (CLI)
python scripts/run_pipeline.py

# 2b. Hoặc mở từng notebook theo thứ tự
jupyter notebook notebooks/01_eda.ipynb
```

## 6. Tham số chính (configs/params.yaml)

| Tham số | Giá trị |
|---------|---------|
| `seed` | 42 |
| `test_size` | 0.2 |
| `cv_folds` | 5 |
| `smote` | True |
| `scaler` | StandardScaler |
| `model_names` | LR, DT, RF, SVM, KNN, GB, XGBoost |
| `semi_supervised.label_ratios` | [0.05, 0.10, 0.20, 0.30] |
| `mining.min_support` | 0.1 |
| `mining.min_confidence` | 0.6 |
| `regression.target_col` | trestbps |
| `regression.models` | LinearRegression, Ridge, XGBRegressor |
