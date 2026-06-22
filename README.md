# Customer Churn Prediction — Supervised Classification

## Project Overview
This project builds and evaluates supervised classification models to predict customer churn.  
Two algorithms are compared: **Logistic Regression** vs **Random Forest**.

---

## Repository Structure

```
churn_classification/
├── churn_classification_notebook.py   # Main notebook (Python script)
├── churn_dataset.csv                  # Generated dataset (1000 rows × 10 cols)
├── metrics_summary.csv                # Final metrics table
├── plots/
│   ├── eda_plots.png                  # EDA visualizations
│   ├── model_comparison.png           # Metrics bar chart + ROC curves
│   ├── confusion_matrices.png         # Confusion matrices (both models)
│   └── feature_importance.png         # Random Forest feature importances
├── Model_Selection_Report.docx        # Full report document
└── README.md                          # This file
```

---

## Environment Setup

### Requirements
- Python 3.8 or higher
- pip package manager

### Step 1 — Create Virtual Environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 2 — Install Dependencies
```bash
pip install scikit-learn==1.4.0 pandas numpy matplotlib seaborn
```

Or with a single command:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Step 3 — Run the Notebook
```bash
cd churn_classification
python churn_classification_notebook.py
```

This will:
1. Generate the synthetic churn dataset
2. Produce all EDA plots (saved to `plots/`)
3. Train and evaluate both models with cross-validation
4. Print metrics summary to console
5. Save all chart images

---

## Running in Jupyter (Optional)

If you prefer an interactive Jupyter notebook:
```bash
pip install jupyter notebook
jupyter notebook
```
Then copy each `# ── Cell N ──` section into separate notebook cells.

---

## Key Results Summary

| Model                | Accuracy | Precision | Recall | F1    | ROC-AUC | CV F1  |
|----------------------|----------|-----------|--------|-------|---------|--------|
| Logistic Regression  | 0.8950   | 0.8519    | 0.7797 | 0.8142| 0.9675  | 0.8311 |
| Random Forest        | 0.9050   | 0.9000    | 0.7627 | 0.8257| 0.9392  | 0.7920 |

**Selected Model:** Random Forest (highest F1 and Accuracy on test set)

---

## Dataset Description

Synthetic dataset simulating telecom customer churn (1000 rows):

| Feature            | Type    | Description                          |
|--------------------|---------|--------------------------------------|
| tenure             | int     | Months as a customer (1–72)          |
| monthly_charges    | float   | Monthly bill amount ($20–$120)       |
| total_charges      | float   | Total billed to date ($50–$8000)     |
| num_products       | int     | Number of subscribed products (1–5)  |
| tech_support       | binary  | Has tech support? (0=No, 1=Yes)      |
| online_security    | binary  | Has online security? (0=No, 1=Yes)   |
| contract_type      | int     | 0=month-to-month, 1=1yr, 2=2yr      |
| paperless_billing  | binary  | Paperless billing? (0=No, 1=Yes)     |
| payment_method     | int     | Payment method category (0–3)        |
| **churn**          | binary  | **Target: churned? (0=No, 1=Yes)**   |

Churn rate: ~29.7%

---

## Methodology

1. **Data Preprocessing** — no missing values; StandardScaler applied for Logistic Regression
2. **Train/Test Split** — 80/20 stratified split
3. **Cross-Validation** — 5-Fold Stratified K-Fold on training data
4. **Algorithms** — Logistic Regression (Pipeline with scaler) and Random Forest
5. **Metrics** — Accuracy, Precision, Recall, F1, ROC-AUC
6. **Visualization** — EDA plots, ROC curves, confusion matrices, feature importance

---

## GitHub Repository

> Push to GitHub with:
> ```bash
> git init
> git add .
> git commit -m "Initial commit: churn classification project"
> git remote add origin https://github.com/<Hima-bindhu123>/churn-classification.git
> git push -u origin main
> ```
