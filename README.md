# 🍄 Mushroom Classification: Edible vs Poisonous

A machine learning project that predicts if a mushroom is **edible (`e`)** or **poisonous (`p`)** from its physical features.
Built for a Kaggle competition dataset.

---

## 📌 Overview

| Item | Details |
|------|---------|
| Task | Binary classification (edible / poisonous) |
| Data | Kaggle competition dataset (`train.csv`, `test.csv`, `sample_submission.csv`) |
| Final model | Tuned LightGBM |
| Validation | 5-fold Stratified Cross-Validation |
| Result | ~100% cross-validation accuracy |

---

## 🔄 Workflow

1. **Load data**: read train, test and sample submission files
2. **Quick look**: check columns and basic stats
3. **Clean data**
   - Fill missing text values with `"missing"` (missing `odor` is linked with the class, so it is useful)
   - Remove duplicate rows
   - Check outliers (all values are real, so nothing is removed)
4. **Visualize**: class balance and odor vs class
5. **Preprocessing pipeline**
   - Drop `habitat` (no common values in train and test) and `veil-type` (same value in every row)
   - One-hot encode text features
   - Fill missing numbers with the median and scale them
6. **Compare 5 models** with 5-fold cross-validation
7. **Tune LightGBM** with GridSearchCV
8. **Final model**: check accuracy, classification report and confusion matrix
9. **Make `submission.csv`** for Kaggle

---

## 🤖 Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM (best, tuned)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, XGBoost, LightGBM

---

## 📁 Project Files

```
├── mushroom_classification_simple.ipynb   # Main notebook
├── README.md                              # This file
└── submission.csv                         # Output predictions (created after running)
```

---

## ▶️ How to Run

**Option 1: Kaggle (easiest)**
1. Upload the notebook to Kaggle.
2. Add the competition dataset to the notebook.
3. Click **Run All**.

**Option 2: On your computer**
1. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
   ```
2. Download `train.csv`, `test.csv` and `sample_submission.csv` from Kaggle.
3. In step 1 of the notebook, change `path` to the folder where your CSV files are.
4. Run all cells in Jupyter Notebook.

---

## 💡 Key Learnings

- Missing values can carry useful information, so filling them with a category can be better than deleting them.
- A preprocessing **Pipeline** keeps train and test data handled in the same way.
- Features that do not match between train and test (like `habitat` here) should be dropped.
- Many models reach almost 100% on this dataset because strong features like `odor` separate the classes very well.
