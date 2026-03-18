# Case Study: Du Doan Benh Tim voi Data Mining (UCI Heart Disease)

## Thong tin Nhom
- Nhom: Nhom 02
- Thanh vien:
- Nguyen Hoa Binh
- (Bo sung cac thanh vien con lai neu co)
- Chu de: Du doan nguy co benh tim
- Dataset: UCI Heart Disease

## Muc tieu
Muc tieu cua nhom la:
> Xay dung he thong phan tich va du doan benh tim tu du lieu lam sang, so sanh hieu qua nhieu mo hinh hoc may, ket hop khai pha tri thuc va trien khai dashboard Streamlit de truc quan ket qua.

Muc tieu cu the:
- Xay dung pipeline end-to-end: load data -> tien xu ly -> modeling -> danh gia -> bao cao.
- Danh gia mo hinh theo bo chi so phu hop y te: Accuracy, Precision, Recall, F1-Score, AUC-ROC, PR-AUC.
- Uu tien giam False Negative de han che bo sot benh nhan co nguy co.
- Thu nghiem them huong semi-supervised va hoi quy de mo rong phan tich.

## 1. Y tuong va Feynman Style
Giai thich de hieu:
- Bai toan can tra loi: "Benh nhan nay co kha nang mac benh tim khong?"
- Dau vao la cac chi so co ban khi kham: tuoi, gioi tinh, huyet ap, cholesterol, ECG, trieu chung...
- Mo hinh hoc tu du lieu qua khu co nhan de du doan cho ca moi.
- Neu mo hinh du doan som nguoi co nguy co cao, bac si co them thong tin de uu tien theo doi.

Tai sao cach tiep can nay phu hop:
- Du lieu co nhan ro rang (`num`).
- So luong mau du de thu nghiem nhieu mo hinh va so sanh.
- Co the danh gia dinh luong bang metric tren tap test.

## 2. Quy trinh thuc hien
1) Load va kiem tra schema du lieu goc  
2) Lam sach va tien xu ly (missing, encode, scale, split, SMOTE)  
3) Feature engineering va feature importance  
4) Khai pha du lieu (Apriori, Clustering, Anomaly Detection)  
5) Huan luyen supervised models + GridSearchCV  
6) Thu nghiem semi-supervised (5/10/20/30% nhan)  
7) Danh gia va visualization (ROC, PR, confusion matrix, error analysis)  
8) Hoi quy chi so huyet ap (`trestbps`)  
9) Trien khai dashboard Streamlit tong hop ket qua

## 3. Mo ta dataset va tien xu ly
### 3.1 Nguon du lieu
- Ten bo du lieu: UCI Heart Disease
- Nguon: https://archive.ics.uci.edu/ml/datasets/heart+disease
- File su dung: data/raw/heart_disease_uci.csv
- Quy mo: 920 mau, 16 cot (bao gom target)

### 3.2 Data Dictionary
| Cot | Mo ta |
|---|---|
| id | Ma dinh danh benh nhan |
| age | Tuoi |
| sex | Gioi tinh |
| dataset | Nguon benh vien con |
| cp | Loai dau nguc |
| trestbps | Huyet ap luc nghi (mmHg) |
| chol | Cholesterol huyet thanh (mg/dL) |
| fbs | Duong huyet luc doi > 120 mg/dL |
| restecg | Ket qua ECG luc nghi |
| thalch | Nhip tim toi da dat duoc |
| exang | Dau nguc khi gang suc |
| oldpeak | ST depression do gang suc |
| slope | Do doc doan ST |
| ca | So mach vanh duoc nhuom mau |
| thal | Ket qua thalassemia |
| num | Target goc 0-4 |

Target su dung trong phan lop:
- `num = 0` -> khong benh
- `num > 0` -> co benh

### 3.3 Cac buoc tien xu ly
- Xu ly missing values theo cot.
- Nhi phan hoa target.
- Encode bien phan loai.
- Chia train/test voi `test_size = 0.2`, `seed = 42`.
- Chuan hoa bien so bang StandardScaler.
- Can bang lop train bang SMOTE.

Thong so cau hinh chinh nam trong file: configs/params.yaml

## 4. Mo hinh va tham so
### 4.1 Mo hinh supervised
- LogisticRegression
- DecisionTree
- RandomForest
- SVM
- KNN
- GradientBoosting
- XGBoost

### 4.2 Cross-validation va toi uu
- CV folds: 5
- Toi uu hyperparameter bang GridSearchCV theo tung model.

### 4.3 Mo hinh semi-supervised
- Supervised-only (chi dung mau co nhan)
- Self-Training
- Label Spreading
- Ti le nhan: 5%, 10%, 20%, 30%

### 4.4 Mo hinh hoi quy
- LinearRegression
- Ridge
- XGBRegressor
- Muc tieu hoi quy: `trestbps`

## 5. Truc quan hoa (Visualization)
Cac hinh chinh duoc sinh trong outputs/figures:
- target_distribution.png
- numerical_distributions.png
- categorical_distributions.png
- correlation_matrix.png
- numerical_by_target.png
- model_comparison_bar.png
- roc_curves.png
- pr_curves.png
- confusion_matrices.png
- semi_supervised_learning_curve.png
- regression_actual_vs_pred.png
- regression_residuals.png
- regression_feature_importance.png
- clustering_results.png
- anomaly_detection.png

## 6. Ket qua chi tiet
### 6.1 So sanh supervised models
Du lieu tu outputs/tables/model_comparison.csv

| Mo hinh | Accuracy | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| GradientBoosting | 0.8533 | 0.8641 | 0.8725 | 0.8683 | 0.9058 | 0.9068 |
| SVM | 0.8424 | 0.8349 | 0.8922 | 0.8626 | 0.9150 | 0.9265 |
| XGBoost | 0.8424 | 0.8544 | 0.8627 | 0.8585 | 0.8996 | 0.8937 |
| RandomForest | 0.8315 | 0.8381 | 0.8627 | 0.8502 | 0.9218 | 0.9354 |
| KNN | 0.8261 | 0.8723 | 0.8039 | 0.8367 | 0.8990 | 0.9033 |
| LogisticRegression | 0.8098 | 0.8252 | 0.8333 | 0.8293 | 0.8868 | 0.9010 |
| DecisionTree | 0.7880 | 0.8119 | 0.8039 | 0.8079 | 0.7970 | 0.7746 |

Nhan xet nhanh:
- Best theo F1: GradientBoosting (0.8683)
- Best theo Recall: SVM (0.8922)
- Best theo PR-AUC: RandomForest (0.9354)

### 6.2 Ket qua semi-supervised
Du lieu tu outputs/tables/semi_supervised_results.csv

Tong quan:
- Supervised-only cho ket qua on dinh va thuong tot hon hai huong semi-supervised.
- Self-Training co Pseudo-label accuracy dao dong 65.4% -> 72.6%.
- Label Spreading chay rat nhanh nhung metric thap hon.

Diem noi bat:
- Ty le nhan 20%: Supervised-only dat F1 = 0.8597 (cao nhat trong bang semi-supervised).
- Ty le nhan 5%: Self-Training dat F1 = 0.7982, thap hon supervised-only (0.8357).

### 6.3 Ket qua hoi quy
Du lieu tu outputs/tables/regression_results.csv

| Mo hinh | CV MAE mean | CV RMSE mean | Train MAE | Train RMSE |
|---|---:|---:|---:|---:|
| LinearRegression | 13.6911 | 17.9915 | 13.2168 | 17.5144 |
| Ridge | 13.6866 | 17.9879 | 13.2162 | 17.5144 |
| XGBRegressor | 15.5560 | 20.7926 | 5.4434 | 7.7925 |

Nhan xet:
- Ridge va LinearRegression generalize tot (train va CV gan nhau).
- XGBRegressor co dau hieu overfit ro (train rat tot, CV xau).

## 7. Insight tu ket qua
Insight #1: GradientBoosting la lua chon can bang tot nhat theo F1.  
Insight #2: Neu uu tien giam bo sot benh nhan, SVM co Recall cao nhat.  
Insight #3: PR-AUC cao cua RandomForest cho thay kha nang phan biet tot tren nhieu nguong.  
Insight #4: Semi-supervised khong vuot supervised-only tren bo du lieu hien tai.  
Insight #5: Khuynh huong overfit xuat hien o XGBRegressor trong bai toan hoi quy.

## 8. Ket luan va de xuat
Ket luan:
- Bai toan du doan benh tim dat ket qua tot voi nhieu mo hinh supervised.
- He thong da hoan thien tu xu ly du lieu den visualization va dashboard demo.

De xuat cai thien tiep:
- Threshold tuning theo muc tieu y te (uu tien Recall/FN).
- Thu nghiem calibration xac suat.
- Ensemble model theo weighted voting/stacking.
- Bo sung feature domain-specific neu co metadata lam sang.
- Danh gia bo sung tren external dataset neu thu thap duoc.

## 9. Huong dan chay du an
### 9.1 Cai dat
```bash
pip install -r requirements.txt
```

### 9.2 Chay pipeline tong
```bash
python scripts/run_pipeline.py
```

### 9.3 Chay notebook theo thu tu
- notebooks/01_eda.ipynb
- notebooks/02_preprocess_feature.ipynb
- notebooks/03_mining_or_clustering.ipynb
- notebooks/04_modeling.ipynb
- notebooks/04b_semi_supervised.ipynb
- notebooks/05_evaluation_report.ipynb
- notebooks/06_regression.ipynb

### 9.4 Chay Streamlit
```bash
streamlit run app.py
```
Mac dinh: http://localhost:8501

## 10. Mo ta dashboard Streamlit
Dashboard trong file app.py gom 7 trang:
- Tong quan du an
- Kham pha du lieu (EDA)
- Khai pha du lieu
- Mo hinh phan loai
- Hoc ban giam sat
- Mo hinh hoi quy
- Du doan truc tuyen

Tinh nang chinh:
- Hien thi bang metric va bieu do tong hop.
- Loc luat ket hop theo support/confidence/lift.
- So sanh semi-supervised theo tung ty le nhan.
- Demo du doan online voi model da train.

## 11. Cau truc thu muc
```
BTL-DATAMINING/
|-- app.py
|-- configs/
|   `-- params.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocess_feature.ipynb
|   |-- 03_mining_or_clustering.ipynb
|   |-- 04_modeling.ipynb
|   |-- 04b_semi_supervised.ipynb
|   |-- 05_evaluation_report.ipynb
|   `-- 06_regression.ipynb
|-- outputs/
|   |-- figures/
|   |-- tables/
|   `-- models/
|-- scripts/
|   `-- run_pipeline.py
|-- src/
|   |-- data/
|   |-- features/
|   |-- mining/
|   |-- models/
|   |-- evaluation/
|   `-- visualization/
|-- requirements.txt
`-- README.md
```

## 12. Link code, notebook, slide
- Repo: https://github.com/hoabinh04/Nhom02_DuDoanBenhTim
- Nhanh den notebook: thu muc notebooks/
- Link slide: (bo sung)

## 13. Luu y
- Day la he thong ho tro hoc tap va phan tich du lieu.
- Ket qua khong thay the chan doan y khoa chuyen nghiep.
