# Advanced Classification

Advanced classification techniques using ensemble methods and optimization strategies.

## 📋 Project Information

- **Notebook:** `Classification Titanic Dataset.ipynb`
- **Topic:** Supervised Learning - Advanced Classification
- **Dataset:** Titanic Dataset
- **Algorithms:** Ensemble methods (Random Forest, Gradient Boosting)
- **Complexity:** Intermediate to Advanced

## 🎯 Learning Objectives

In this project, you will learn:
- Advanced classification techniques
- Ensemble methods untuk classification
- Hyperparameter tuning
- Cross-validation untuk robust evaluation
- Feature importance analysis

## 📊 Dataset

**Titanic Dataset** - Binary classification problem untuk prediksi keselamatan penumpang

- **Size:** ~891 records
- **Features:** 11 columns
- **Target:** Survived (0=No, 1=Yes)
- **Type:** Binary Classification

### Features Detail:
| Feature | Tipe | Deskripsi |
|---------|------|-----------|
| PassengerId | int | ID penumpang |
| Pclass | int | Kelas tiket (1, 2, 3) |
| Name | str | Nama penumpang |
| Sex | str | Jenis kelamin (male, female) |
| Age | float | Umur dalam tahun |
| SibSp | int | Jumlah saudara/suami di kapal |
| Parch | int | Jumlah orang tua/anak di kapal |
| Ticket | str | Nomor tiket |
| Fare | float | Harga tiket |
| Cabin | str | Nomor kabin |
| Embarked | str | Port keberangkatan (C, Q, S) |

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation & analysis |
| **numpy** | Numerical computing |
| **matplotlib** | Plotting & visualization |
| **seaborn** | Statistical visualization |
| **scikit-learn** | ML algorithms & metrics |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook Classification\ Titanic\ Dataset.ipynb
```

### Google Colab
1. Upload file ke Colab
2. Run cells dari atas ke bawah
3. Download results jika diperlukan

## 📝 Isi Notebook

### 1. **Setup & Import**
Mengimpor semua library yang diperlukan

### 2. **Data Loading**
- Load dataset Titanic
- Display basic information
- Check shape dan columns

### 3. **Exploratory Data Analysis**
- Data shape & info
- Missing values analysis
- Statistical summary
- Correlation analysis
- Data visualization:
  - Distribution plots
  - Count plots
  - Box plots
  - Heatmaps

### 4. **Data Cleaning & Preprocessing**
- Handle missing values:
  - Age: Mean/Median imputation
  - Embarked: Mode imputation
  - Cabin: Drop atau create feature
- Drop irrelevant columns (Ticket, Cabin, Name, PassengerId)
- Encode categorical variables:
  - Sex: Male/Female → 1/0
  - Embarked: One-hot encoding

### 5. **Feature Engineering**
- Create new features:
  - Family size (SibSp + Parch)
  - Is alone (Family size == 1)
  - Title extraction dari Name
- Feature scaling (StandardScaler)

### 6. **Train-Test Split**
- 80% training, 20% testing
- Stratified split untuk balanced distribution

### 7. **Model Training**
Multiple classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (optional)

### 8. **Model Evaluation**
- Accuracy score
- Confusion matrix
- Classification report
- ROC-AUC score
- Cross-validation score

### 9. **Model Comparison**
- Compare performance across models
- Visualize results
- Select best model

### 10. **Feature Importance**
- Extract feature importance dari tree-based models
- Visualize top features
- Interpret results

### 11. **Hyperparameter Tuning** (optional)
- Grid search untuk optimal parameters
- Cross-validation
- Final model training

### 12. **Final Predictions & Conclusion**
- Make predictions on test set
- Final evaluation
- Summary & recommendations

## 📈 Output yang Dihasilkan

- **Model Comparison Table** - Performance metrics
- **Confusion Matrices** - Visual & numerical
- **Classification Reports** - Detailed metrics
- **ROC Curves** - Performance comparison
- **Feature Importance Plots** - Top contributing features
- **Sample Predictions** - Example predictions with probabilities

## 💡 Key Concepts

### Binary Classification
Prediksi satu dari dua kemungkinan kategori

### Confusion Matrix
```
                 Predicted 1  Predicted 0
Actual 1           TP          FN
Actual 0           FP          TN
```

### Important Metrics

1. **Accuracy:** (TP+TN)/Total
   - Keseluruhan correctness
   - Baik untuk balanced data

2. **Precision:** TP/(TP+FP)
   - Accuracy dari positive predictions
   - Penting ketika false positives costly

3. **Recall:** TP/(TP+FN)
   - Coverage dari actual positives
   - Penting ketika false negatives costly

4. **F1-Score:** Harmonic mean dari precision & recall
   - Balanced metric
   - Baik untuk imbalanced data

5. **ROC-AUC:** Area under ROC curve
   - Threshold-independent metric
   - 0.5 = random, 1.0 = perfect

### Feature Engineering
Process menciptakan fitur baru dari existing data untuk meningkatkan model performance

## 📥 Kaggle Dataset Download

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API key (dari Kaggle Settings)
# Letakkan kaggle.json di ~/.kaggle/

# Download Titanic dataset
kaggle competitions download -c titanic

# Or download dataset version
kaggle datasets download -d brendan45774/test-file
```

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Missing values | Use imputation or drop rows |
| Categorical variables | One-hot encoding or label encoding |
| Class imbalance | Use class_weight='balanced' or resampling |
| Overfitting | Use cross-validation, regularization, dropout |
| Low accuracy | Feature engineering, hyperparameter tuning, ensemble methods |

## 💻 Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## 📚 Referensi

- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.classification)
- [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic)
- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- [ROC Curve Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Feature Engineering Guide](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

## ✅ Checklist

- [ ] Load & explore dataset
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Feature engineering
- [ ] Train-test split
- [ ] Train multiple models
- [ ] Evaluate models
- [ ] Compare performance
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning (optional)
- [ ] Final predictions

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete
