# Ensemble Methods - Random Forest Classification

Apply ensemble learning techniques to real-world multi-class classification problems using Random Forest.

## 📋 Project Information

- **Notebook:** `Random Forest - Accident Prediction.ipynb`
- **Topic:** Supervised Learning - Ensemble Methods
- **Dataset:** Accident Severity Dataset
- **Model:** Random Forest Classifier
- **Complexity:** Intermediate to Advanced

## 🎯 Learning Objectives

In this project, you will:
- Mengaplikasikan seluruh konsep machine learning yang telah dipelajari
- Melakukan end-to-end data analysis & modeling
- Feature engineering & feature selection
- Ensemble methods (Random Forest)
- Model evaluation & optimization
- Interpretasi dan komunikasi hasil

## 📊 Dataset

**Accident Dataset** (dataset_kecelakaan.csv) - Dataset tentang kecelakaan jalan

- **Features:** Berbagai variabel yang mempengaruhi kecelakaan (jenis kendaraan, waktu, cuaca, jalan, dll)
- **Target:** Severitas kecelakaan (e.g., minor, serious, fatal)
- **Type:** Multi-class classification
- **Size:** Tergantung dataset

### Expected Features (example):
- Time/Date (jam, hari, musim)
- Location (jalan, kota, area)
- Weather conditions (hujan, siang/malam)
- Vehicle type (mobil, motor, truck)
- Number of vehicles involved
- Number of injuries/fatalities
- Road conditions (licin, rusak, dll)

### Target Variable:
- Severity level (e.g., 0=minor, 1=moderate, 2=serious, 3=fatal)
- Atau binary: injury vs no injury

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation & analysis |
| **numpy** | Numerical operations |
| **matplotlib** | Visualization |
| **seaborn** | Statistical visualization |
| **scikit-learn** | ML algorithms & evaluation |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook Random\ Forest\ -\ Accident\ Prediction.ipynb
```

### Google Colab
1. Upload dataset & notebook
2. Run cells dari atas ke bawah
3. Export hasil jika diperlukan

## 📝 Inti Ujian

### 1. **Problem Statement & Dataset Loading**
- Definisikan problem dengan jelas
- Load dataset
- Initial exploration

### 2. **Exploratory Data Analysis (EDA)**
- Data shape, info, missing values
- Statistical summary
- Visualisasi distribusi
- Correlation analysis
- Target variable distribution

### 3. **Data Cleaning & Preprocessing**
- Handle missing values (imputation/dropping)
- Detect & handle outliers (optional)
- Encode categorical variables
- Handle data types

### 4. **Feature Engineering**
- Create new features dari existing ones
- Extract temporal features (from date)
- Domain-specific features
- Feature encoding

### 5. **Feature Selection**
Metode untuk select most important features:

**Method 1: SelectKBest dengan Chi-square**
```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

**Method 2: Random Forest Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**Method 3: Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
rfe.fit(X, y)
```

### 6. **Data Splitting**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 7. **Feature Scaling** (jika perlu)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 8. **Model Training - Multiple Models**

#### Random Forest (Primary)
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
```

#### Other Models untuk Comparison:
- Logistic Regression
- Decision Tree
- Gradient Boosting
- SVM (optional)

### 9. **Model Evaluation**

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

y_pred = rf_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print(classification_report(y_test, y_pred))
```

### 10. **Cross-Validation**
```python
from sklearn.model_selection import cross_validate

cv_scores = cross_validate(
    rf_model, X_train, y_train, 
    cv=5, 
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
)

for metric, scores in cv_scores.items():
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 11. **Hyperparameter Tuning** (optional)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 12. **Visualization**

**Feature Importance Plot:**
```python
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()
```

**Confusion Matrix Heatmap:**
```python
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
```

**Model Comparison:**
```python
models_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree', ...],
    'Accuracy': [...],
    'Precision': [...],
    'Recall': [...],
    'F1-Score': [...]
})

models_comparison.set_index('Model').plot(kind='bar')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()
```

### 13. **Insights & Interpretation**
- Key findings dari analysis
- Which features matter most?
- Model performance interpretation
- Limitations & future improvements
- Actionable recommendations

### 14. **Final Model & Conclusion**
- Pilih best model berdasarkan evaluation
- Summary dari key results
- Recommendations untuk accident prevention
- Future work suggestions

## 📈 Expected Output

- **EDA Visualizations** - Distributions, correlations, relationships
- **Feature Importance** - Top contributing features
- **Confusion Matrix** - Prediction analysis
- **Classification Report** - Detailed metrics per class
- **Model Comparison** - Performance across models
- **Cross-Validation Scores** - Robust performance estimates
- **Summary Report** - Key findings & recommendations

## 💡 Key Concepts

### Random Forest
Ensemble method yang kombinasi multiple decision trees:
- **Advantages:**
  - High accuracy
  - Handles non-linear relationships
  - Built-in feature importance
  - Robust ke outliers
  - No feature scaling needed
- **Disadvantages:**
  - Computationally expensive
  - Black-box (less interpretable)
  - Can overfit dengan default params

### Random Forest Parameters

| Parameter | Deskripsi |
|-----------|-----------|
| `n_estimators` | Number of trees (100, 200, 300) |
| `max_depth` | Tree depth (10-20 typical) |
| `min_samples_split` | Min samples untuk split (2-10) |
| `min_samples_leaf` | Min samples per leaf (1-5) |
| `max_features` | Features per split ('sqrt', 'log2', None) |
| `random_state` | Reproducibility |

### Classification Metrics

- **Accuracy:** Overall correctness
- **Precision:** Of predicted positive, how many are true positive
- **Recall:** Of actual positive, how many did we find
- **F1-Score:** Harmonic mean dari precision & recall
- **Weighted Average:** Account untuk class imbalance

### Feature Selection Benefits
- Reduce model complexity
- Faster training/inference
- Better generalization
- Interpretability
- Cost reduction

## ⚠️ Common Mistakes & How to Avoid

| Mistake | Solution |
|---------|----------|
| Data leakage | Don't scale before split, don't use test data untuk feature selection |
| Class imbalance ignored | Use stratified split, weighted metrics, resampling |
| No cross-validation | Always use CV untuk robust estimates |
| Overfitting | Monitor train vs test performance, use regularization |
| Poor feature engineering | Domain knowledge, create meaningful features |
| Not exploring data enough | Spend time dengan EDA, understand patterns |

## 💻 Complete Pipeline Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load & Explore
df = pd.read_csv('dataset_kecelakaan.csv')
print(df.info())
print(df.describe())

# 2. Handle missing values
df = df.dropna()

# 3. Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 4. Separate features & target
X = df.drop('target', axis=1)
y = df['target']

# 5. Feature selection
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Train
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# 8. Evaluate
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

## 📚 Referensi

- [Random Forest - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Random Forest Explained](https://en.wikipedia.org/wiki/Random_forest)

## ✅ Rubric & Grading Criteria

- **EDA & Data Cleaning:** 20%
- **Feature Engineering & Selection:** 20%
- **Model Training & Evaluation:** 30%
- **Cross-Validation & Hyperparameter Tuning:** 15%
- **Visualization & Interpretation:** 10%
- **Report Quality & Insights:** 5%

## ✅ Checklist

- [ ] Load & explore dataset
- [ ] Handle missing values & outliers
- [ ] Encode categorical variables
- [ ] Perform EDA with visualizations
- [ ] Select relevant features
- [ ] Split train/test
- [ ] Train Random Forest model
- [ ] Compare dengan other models
- [ ] Perform cross-validation
- [ ] Evaluate dengan multiple metrics
- [ ] Hyperparameter tuning (optional)
- [ ] Visualize results
- [ ] Write summary report
- [ ] Document insights & recommendations

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ UTS Submission

## 📝 Submission Notes

- Ensure semua cells runnable tanpa error
- Explain setiap step dengan comments/markdown
- Include visualizations yang informatif
- Provide clear conclusions & recommendations
- Submit dalam format .ipynb
