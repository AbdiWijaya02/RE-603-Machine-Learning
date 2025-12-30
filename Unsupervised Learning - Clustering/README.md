# Unsupervised Clustering Analysis

Discover patterns and group similar data points using K-Means clustering and optimization techniques.

## 📋 Project Information

- **Notebook:** `Unsupervised Learning - Clustering.ipynb`
- **Topic:** Unsupervised Learning - Clustering
- **Dataset:** Biometric Dataset (Height and Weight)
- **Algorithm:** K-Means, Elbow Method, Silhouette Analysis
- **Complexity:** Beginner to Intermediate

## 🎯 Learning Objectives

In this project, you will learn:
- Memahami konsep clustering dalam unsupervised learning
- Implementasi K-Means algorithm
- Menentukan optimal number of clusters (Elbow Method)
- Silhouette analysis untuk cluster evaluation
- Visualisasi hasil clustering
- Interpretasi dan karakterisasi cluster

## 📊 Dataset

**Height and Weight Dataset** (berat_tinggi.csv) - Dataset body measurements untuk clustering

- **Fitur:**
  - `berat` (weight) - dalam kilogram
  - `tinggi` (height) - dalam centimeter
- **Size:** Tergantung dataset yang digunakan
- **Type:** Unsupervised (tidak ada label)

### Dataset Characteristics:
- Numerical features
- No missing values (assumed clean)
- 2D dataset (mudah untuk visualisasi)

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation & loading |
| **numpy** | Numerical operations |
| **matplotlib** | Plotting |
| **seaborn** | Statistical visualization |
| **scikit-learn** | K-Means & metrics |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook Unsupervised\ Learning\ -\ Clustering.ipynb
```

### Google Colab
1. Upload notebook
2. Upload atau load dataset
3. Run cells from top to bottom

## 📝 Isi Notebook

### 1. **Import Library**
Mengimpor semua dependencies

### 2. **Load Dataset**
- Load CSV file
- Display shape & head
- Check data types

### 3. **Exploratory Data Analysis**
- Data shape & info
- Statistical summary
- Data distribution visualization:
  - Histogram untuk berat & tinggi
  - Scatter plot untuk sebaran data
- Correlation analysis

### 4. **Data Preprocessing**
- Check missing values
- Data cleaning (jika ada outliers)
- Feature scaling (StandardScaler/MinMaxScaler):
  - Penting karena K-Means distance-based
  - Sama-ratakan skala fitur

### 5. **Elbow Method**
- Train K-Means dengan k = 1 to 10
- Hitung inertia (within-cluster sum of squares)
- Plot inertia vs number of clusters
- Identifikasi "elbow point" sebagai optimal k
- Interpretasi: Mencari k dimana penurunan inertia mulai melambat

### 6. **K-Means Clustering**
- Train final K-Means model dengan optimal k
- Get cluster labels untuk setiap sample
- Get cluster centers (centroids)

### 7. **Silhouette Analysis**
- Hitung silhouette score untuk setiap sample (-1 to 1)
- Interpretasi:
  - Close to 1: Well-clustered
  - Close to 0: Borderline
  - Negative: Assigned to wrong cluster
- Plot silhouette diagram
- Calculate average silhouette score

### 8. **Cluster Visualization**
- 2D scatter plot dengan color per cluster
- Highlight centroids
- Add cluster boundaries (jika feasible)
- Interpretasi visual patterns

### 9. **Cluster Characterization**
- Statistik per cluster:
  - Mean height per cluster
  - Mean weight per cluster
  - Cluster size (jumlah samples)
- Buat cluster profiles/descriptions

### 10. **Results Interpretation**
- Diskusi temuan
- Interpretasi cluster meaning (e.g., short vs tall people)
- Limitations dan future improvements

## 📈 Output yang Dihasilkan

- **Elbow Curve** - Inertia vs number of clusters
- **Silhouette Plot** - Cluster quality visualization
- **Cluster Scatter Plot** - 2D visualization dengan centroids
- **Cluster Statistics** - Mean values per cluster
- **Cluster Profiles** - Karakteristik setiap cluster

## 💡 Key Concepts

### Clustering
Teknik unsupervised untuk mengelompokkan data points yang mirip tanpa predefined labels

### K-Means Algorithm

**Algoritma:**
1. Initialize k random centroids
2. Assign each point ke nearest centroid
3. Update centroids sebagai mean dari assigned points
4. Repeat steps 2-3 until convergence

**Mathematical:**
- Distance: Euclidean distance
- Objective: Minimize within-cluster variance (inertia)

$$Inertia = \sum_{i=0}^{n} \min_{\mu_j \in C} ||x_i - \mu_j||^2$$

### Elbow Method
Teknik untuk menentukan optimal number of clusters:
- Plot inertia vs k
- Cari "elbow" point dimana penurunan inertia mulai flatten
- Trade-off antara model complexity dan error reduction

### Silhouette Score

Mengukur quality dari clustering:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Dimana:
- $a(i)$ = mean distance ke points dalam cluster yang sama
- $b(i)$ = mean distance ke points dalam nearest cluster

**Interpretasi:**
- Range: -1 to 1
- Score > 0.5: Well-separated clusters
- Score < 0.25: Overlapping clusters

### Scaling Importance
K-Means menggunakan distance-based:
- Features dengan range besar akan dominate
- Perlu normalize/scale semua features

## 📥 Dataset Download

Dataset height and weight bisa diperoleh dari:

```bash
# Jika dataset di folder
# Letakkan berat_tinggi.csv di working directory

# Atau download dari Kaggle
kaggle datasets download -d [dataset-name]
```

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Elbow unclear | Try different k range, larger visualization |
| Bad silhouette score | Try different k, or scale data better |
| Inconsistent results | Set random_state for reproducibility |
| Outliers affecting results | Perform outlier detection/removal before clustering |

## 💻 Code Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Optimal K-Means
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Silhouette score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

## 📚 Referensi

- [K-Means Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Elbow Method Explanation](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- [Silhouette Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [Clustering Evaluation Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [Unsupervised Learning Guide](https://machinelearningmastery.com/unsupervised-learning/)

## ✅ Checklist

- [ ] Load dataset
- [ ] EDA & visualization
- [ ] Feature scaling
- [ ] Elbow method analysis
- [ ] Optimal k determination
- [ ] Train K-Means
- [ ] Silhouette analysis
- [ ] Cluster visualization
- [ ] Cluster characterization
- [ ] Interpretation & conclusions

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete
