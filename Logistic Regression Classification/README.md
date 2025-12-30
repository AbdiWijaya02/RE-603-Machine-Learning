# Logistic Regression Classification

Master binary classification using logistic regression with comprehensive evaluation metrics.

## 📋 Project Information

- **Notebook:** `Logistic Regression Classification.ipynb`
- **Topic:** Supervised Learning - Logistic Regression
- **Dataset:** Titanic Dataset
- **Algorithm:** Logistic Regression
- **Complexity:** Beginner to Intermediate

## 🎯 Learning Objectives

In this project, you will learn:
- Memahami algoritma Logistic Regression
- Implementasi Logistic Regression dari scikit-learn
- Interpretasi model coefficients
- Confusion matrix dan classification metrics
- Threshold adjustment untuk optimization

## 📊 Dataset

**Titanic Dataset** - Dataset penumpang Titanic untuk binary classification

- **Fitur:** PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Target:** Survived (0 atau 1)
- **Tipe:** Binary Classification
- **Size:** ~891 samples

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation |
| **numpy** | Numerical operations |
| **matplotlib** | Visualization |
| **seaborn** | Statistical visualization |
| **scikit-learn** | Logistic Regression & metrics |

## 🚀 Cara Menjalankan

### Di Local Machine
```bash
jupyter notebook Logistic\ Regression\ Classification.ipynb
```

### Di Google Colab
1. Upload notebook
2. Jalankan cells
3. Dataset akan auto-load dari Colab built-in

## 📝 Isi Notebook

### 1. **Import Library**
Loading semua dependencies

### 2. **Load & Explore Dataset**
- Data loading
- Data shape dan info
- Missing values check
- Basic statistics

### 3. **Data Preprocessing**
- Handling missing values
- Encoding categorical variables (Sex, Embarked)
- Feature selection
- Train-test split (80-20)
- Feature scaling (StandardScaler/MinMaxScaler)

### 4. **Model Training**
- Membuat Logistic Regression model
- Fitting model dengan training data
- Interpretasi coefficients

### 5. **Model Prediction**
- Prediksi pada test set
- Probability predictions
- Adjusting decision threshold

### 6. **Model Evaluation**
- **Confusion Matrix:** TP, TN, FP, FN
- **Classification Report:** Precision, Recall, F1-Score per class
- **Accuracy Score:** Overall accuracy
- **ROC-AUC Score:** Model discrimination ability
- **ROC Curve:** Visualisasi trade-off antara TPR dan FPR

### 7. **Performance Analysis**
- Per-class performance
- Threshold optimization
- Cross-validation score (jika ada)

### 8. **Kesimpulan**
- Model interpretation
- Feature importance
- Rekomendasi improvements

## 📈 Output yang Dihasilkan

- **Confusion Matrix** - Visual matrix
- **Classification Report** - Detailed metrics
- **Accuracy & AUC** - Model scores
- **ROC Curve** - Performance visualization
- **Feature Coefficients** - Model weights visualization

## 💡 Key Concepts

### Logistic Regression

Persamaan Logistic Function:
$$P(y=1) = \frac{1}{1 + e^{-z}}$$

Dimana:
$$z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ$$

- Output adalah probability (0-1)
- Default threshold = 0.5
- Linear dalam log-odds space

### Classification Metrics

**Confusion Matrix Components:**
- **TP (True Positive):** Predicted 1, Actual 1
- **TN (True Negative):** Predicted 0, Actual 0
- **FP (False Positive):** Predicted 1, Actual 0
- **FN (False Negative):** Predicted 0, Actual 1

**Metrics:**
- **Accuracy:** (TP+TN)/(Total) - Overall correctness
- **Precision:** TP/(TP+FP) - Accuracy of positive predictions
- **Recall:** TP/(TP+FN) - Coverage of actual positives
- **F1-Score:** 2×(Precision×Recall)/(Precision+Recall) - Balanced metric
- **ROC-AUC:** Area under ROC curve (0-1)

### ROC Curve

- X-axis: False Positive Rate (FPR) = FP/(FP+TN)
- Y-axis: True Positive Rate (TPR) = TP/(TP+FN)
- Area Under Curve (AUC):
  - 0.9-1.0 = Excellent
  - 0.8-0.9 = Good
  - 0.7-0.8 = Fair
  - 0.6-0.7 = Poor
  - 0.5 = Random

## 📥 Dataset Download

```bash
# Dari Kaggle
kaggle datasets download -d titanic

# Extract
unzip titanic.zip

# Atau di Google Colab, sudah built-in
```

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Singular matrix error | Check multicollinearity, drop redundant features |
| Class imbalance | Use class_weight='balanced' |
| Poor AUC score | Check feature quality, tune hyperparameters |
| Module error | `pip install --upgrade scikit-learn` |

## 💻 Example Code Snippet

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Create model
model = LogisticRegression(random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba[:, 1])}")
```

## 📚 Referensi

- [Logistic Regression - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Understanding ROC Curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)

## ✅ Checklist

- [ ] Install libraries
- [ ] Load dataset
- [ ] EDA & preprocessing
- [ ] Train Logistic Regression
- [ ] Generate predictions
- [ ] Evaluate dengan confusion matrix
- [ ] Generate classification report
- [ ] Plot ROC curve
- [ ] Analyze results

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete
