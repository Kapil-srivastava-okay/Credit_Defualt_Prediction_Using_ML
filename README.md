# 💳 Credit Card Default Prediction Using Machine Learning

## 📘 Overview

This project aims to predict credit card default using machine learning models on the **Default of Credit Card Clients** dataset. The goal is to identify customers likely to default, helping financial institutions make informed lending decisions.

Three models are evaluated:
- Logistic Regression
- Random Forest
- XGBoost

## 📂 Repository Structure

- `Predicting_Credit_Default_Report.pdf` — Full project report
- `stroke_detection.ipynb` — Jupyter notebook (original name, update if needed)
- `data/` — Dataset files (not included here; available at UCI)
- `README.md` — This file

---

## 🧠 Problem Statement

Predict whether a credit card client will default next month based on past payment behavior and demographic features. This is a binary classification problem (1 = Default, 0 = No Default).

---

## 📊 Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Instances: 30,000
- Features: 23 predictive variables (X1–X23)
- Target: `Y` (default = 1, no default = 0)
- Class Imbalance: 28% default vs 72% no default

### Key Variables
- **X1:** Credit Limit
- **X2–X5:** Gender, Education, Marital Status, Age
- **X6–X11:** Past payment status (last 6 months)
- **X12–X17:** Monthly bill amounts
- **X18–X23:** Monthly payment amounts

---

## ⚙️ Methodology

### 📌 Data Preprocessing
- Dropped header row, renamed index
- One-hot encoded categorical variables (`X2–X4`)
- Standardized numeric features (`X1, X5–X23`)
- Addressed class imbalance using:
  - `class_weight='balanced'`
  - SMOTE (Synthetic Minority Oversampling)

### 📌 Models and Tuning

#### Logistic Regression
- `max_iter=20000`, `class_weight='balanced'`
- Grid search on `C` and solvers
- ROC-AUC ~ 0.72

#### Random Forest
- `n_estimators=200`, `max_depth=20`, `class_weight='balanced'`
- Grid search on `max_features`, `min_samples_split`
- ROC-AUC ~ 0.76

#### XGBoost
- `scale_pos_weight=3.55`, `learning_rate=0.1`, `max_depth=10`
- Grid search on `n_estimators`, `subsample`
- ROC-AUC ~ 0.75

---

## 📈 Results

| Model              | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|-------------------|----------|---------------|------------|--------|----------|
| Logistic Regression | 81.2%   | 0.62          | 0.38       | 0.47   | 0.72     |
| Random Forest       | 81.4%   | 0.62          | 0.40       | 0.49   | 0.76     |
| XGBoost             | 78.5%   | 0.51          | 0.49       | 0.50   | 0.75     |

> 📌 **Best Model**: **Random Forest** due to its superior balance of precision, recall, and AUC.

---

## 🔍 Key Insights

- **Payment history features (X6–X11)** were the most predictive.
- **Bill statement amounts (X12–X17)** were moderately important.
- **Demographics (X2–X5)** were less useful for prediction.
- Random Forest and XGBoost showed robust performance for imbalanced classification tasks.

---

## 📊 Visualizations

- 📈 ROC and Precision-Recall Curves
- 📉 Confusion Matrix
- 📊 Feature Importance (XGBoost & Random Forest)

---

## 🛠️ Technologies Used

- Python (Pandas, Scikit-learn, XGBoost)
- SMOTE from imblearn
- Matplotlib, Seaborn for visualization
- Jupyter Notebook for code and analysis

---

## 🧪 Future Work

- Explore neural networks or stacking ensembles
- Deploy as an API using Flask or Streamlit
- Test model with recent or live financial data
- Model explainability with SHAP or LIME

---

## 👨‍💻 Author

**Kapil Srivastava**  
MSc Data Science | Coventry University  
📧 srivastavk@coventry.ac.uk  
🔗 [[LinkedIn Profile](https://www.linkedin.com/in/kapil-srivastava-730a4916a/)](#)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

