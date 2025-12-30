# Linear Regression Analysis

A comprehensive guide to building and evaluating linear regression models for predictive modeling.

## 📋 Project Information

- **Notebook:** `Regression Analysis.ipynb`
- **Topic:** Supervised Learning - Linear Regression
- **Dataset:** USA Housing Dataset
- **Algorithm:** Linear Regression
- **Complexity:** Beginner to Intermediate

## 🎯 Learning Objectives

In this project, you will learn to:
- Prepare and clean datasets for machine learning
- Perform Exploratory Data Analysis (EDA)
- Build and train linear regression models
- Evaluate models using various metrics
- Perform diagnostic analysis (heteroscedasticity, autocorrelation, normality)

## 📊 Dataset

**USA Housing Dataset** - Contains information about residential property prices in the USA

- **Features:**
  - Avg. Area Income
  - Avg. Area House Age
  - Avg. Area Number of Rooms
  - Avg. Area Number of Bedrooms
  - Area Population
  - Price (Target Variable)

**Sumber:** Built-in Google Colab

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation dan analysis |
| **numpy** | Numerical computing |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical data visualization |
| **scikit-learn** | Machine learning algorithms |
| **scipy** | Statistical functions |
| **statsmodels** | Statistical modeling dan hypothesis testing |

## 🚀 Cara Menjalankan

### Di Local Machine
```bash
jupyter notebook Regression\ Analysis.ipynb
```

### Di Google Colab
1. Upload notebook ke Google Colab
2. Dataset sudah tersedia di built-in Colab
3. Jalankan cell secara berurutan

## 📝 Isi Notebook

### 1. **Import Library**
Mengimpor semua library yang diperlukan untuk analisis

### 2. **Load Dataset**
Membaca dan menampilkan dataset USA Housing

### 3. **Exploratory Data Analysis (EDA)**
- Statistik deskriptif
- Visualisasi distribusi data
- Correlation analysis
- Checking for missing values

### 4. **Data Preprocessing**
- Train-test split (80-20)
- Feature scaling (jika diperlukan)

### 5. **Model Training**
- Membuat Linear Regression model
- Training model dengan data training

### 6. **Model Evaluation**
- Prediksi pada data test
- Menghitung metrik: MAE, MSE, RMSE, R²

### 7. **Diagnostic Analysis**
- **Heteroskedastisitas (Breusch-Pagan Test):** Mengecek apakah varians residual konstan
- **Autokorelasi (Durbin-Watson Test):** Mengecek apakah ada korelasi antar residual
- **Normalitas (Shapiro-Wilk Test):** Mengecek apakah residual terdistribusi normal
- **Visualisasi residual:** Plot residual vs predicted values

### 8. **Kesimpulan dan Rekomendasi**
- Interpretasi hasil model
- Kekuatan dan kelemahan model

## 📈 Output yang Dihasilkan

- **Coefficient values** - Bobot fitur dalam model
- **Performance metrics** - MAE, MSE, RMSE, R² Score
- **Diagnostic plots** - Visualisasi untuk pengecekan asumsi
- **Predictions** - Nilai prediksi harga properti

## 💡 Key Concepts

### Linear Regression
Persamaan: **y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ**

- y = target variable (harga)
- β₀ = intercept
- β₁, β₂, ... = coefficients (slopes)
- x₁, x₂, ... = independent variables (fitur)

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Rata-rata nilai absolut error
- **MSE (Mean Squared Error):** Rata-rata squared error
- **RMSE (Root Mean Squared Error):** Akar dari MSE
- **R² Score:** Proporsi varians yang dijelaskan oleh model (0-1)

### Diagnostic Tests

- **Breusch-Pagan Test:** H₀ = residual homoskedastis
- **Durbin-Watson:** Nilai 2 berarti tidak ada autokorelasi
- **Shapiro-Wilk:** H₀ = residual normally distributed

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run: `pip install [module_name]` |
| Dataset not found | Ensure in Google Colab or download manually |
| Memory error | Use data subset or reduce batch size |

## 📚 Referensi

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Understanding Regression Diagnostics](https://en.wikipedia.org/wiki/Regression_diagnostics)

## ✅ Checklist

- [ ] Install semua dependencies
- [ ] Load dataset dengan sukses
- [ ] Menjalankan EDA
- [ ] Training model
- [ ] Evaluasi model
- [ ] Melakukan diagnostic tests
- [ ] Interpretasi hasil

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete
