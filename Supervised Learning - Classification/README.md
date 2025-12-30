# Classification Pipeline

A comprehensive guide to implementing classification workflows with multiple algorithms.

## 📋 Project Information

- **Notebook:** `Supervised Learning - Classification.ipynb`
- **Topic:** Supervised Learning - Classification
- **Dataset:** Titanic Dataset
- **Algorithms:** Multiple classification models
- **Complexity:** Beginner to Intermediate

## 🎯 Learning Objectives

In this project, you will learn to:
- Memahami konsep classification dalam supervised learning
- Melakukan data preprocessing untuk classification
- Feature engineering dan selection
- Membuat dan training berbagai classification models
- Evaluasi model dengan classification metrics

## 📊 Dataset

**Titanic Dataset** - Dataset tentang penumpang Titanic dengan informasi keselamatan mereka

- **Fitur:**
  - PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Target:** Survived (0 = Tidak selamat, 1 = Selamat)
- **Tipe:** Binary Classification
- **Sumber:** Built-in Google Colab atau Kaggle

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation dan analysis |
| **numpy** | Numerical computing |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical data visualization |
| **scikit-learn** | Classification algorithms |

## 🚀 Cara Menjalankan

### Di Local Machine
```bash
jupyter notebook Supervised\ Learning\ -\ Classification.ipynb
```

### Di Google Colab
1. Upload notebook ke Google Colab
2. Dataset sudah tersedia di built-in Colab
3. Jalankan cell secara berurutan

## 📝 Isi Notebook

### 1. **Import Library**
Mengimpor library yang diperlukan

### 2. **Load Dataset**
Membaca Titanic dataset

### 3. **Exploratory Data Analysis (EDA)**
- Data shape dan info
- Missing values analysis
- Statistical summary
- Data visualization

### 4. **Data Preprocessing**
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split

### 5. **Feature Engineering**
- Feature selection menggunakan SelectKBest
- Chi-square test untuk feature importance
- Dropping irrelevant features

### 6. **Model Training**
- Melatih berbagai classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting (jika ada)

### 7. **Model Evaluation**
- Classification report (Precision, Recall, F1-Score)
- Confusion matrix
- Accuracy score
- ROC-AUC score (jika applicable)

### 8. **Model Comparison**
- Membandingkan performa berbagai model
- Visualisasi hasil

## 📈 Output yang Dihasilkan

- **Classification metrics** - Accuracy, Precision, Recall, F1-Score
- **Confusion matrices** - Prediksi vs actual
- **Feature importance** - Fitur yang paling berpengaruh
- **Model comparison** - Perbandingan antar model
- **Visualizations** - Grafik dan plot hasil analisis

## 💡 Key Concepts

### Classification Problem
Memprediksi kategori/label dari data baru berdasarkan data historis

### Binary Classification vs Multi-class Classification
- **Binary:** 2 kelas (contoh: Selamat/Tidak Selamat)
- **Multi-class:** >2 kelas

### Evaluation Metrics

- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **Precision:** TP / (TP + FP) - Akurasi prediksi positif
- **Recall:** TP / (TP + FN) - Coverage positif actual
- **F1-Score:** Harmonic mean dari Precision dan Recall

**Confusion Matrix:**
```
                 Predicted Positive  Predicted Negative
Actual Positive         TP                   FN
Actual Negative         FP                   TN
```

## 📥 Dataset Download

Dataset Titanic sudah tersedia di Google Colab dan Kaggle:

```bash
# Dari Kaggle
kaggle datasets download -d titanic
unzip titanic.zip
```

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run: `pip install [module_name]` |
| Dataset not found | Download from Kaggle or use Colab |
| Class imbalance warning | Normal - Titanic dataset is imbalanced |
| Memory issues | Use subset or reduce data size |

## 📚 Referensi

- [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix Explanation](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Titanic Dataset Information](https://www.kaggle.com/c/titanic)

## ✅ Checklist

- [ ] Install dependencies
- [ ] Load dan explore dataset
- [ ] Data preprocessing
- [ ] Feature engineering
- [ ] Training multiple models
- [ ] Model evaluation
- [ ] Model comparison
- [ ] Interpretasi hasil

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete
