# Machine Learning Projects Collection

Comprehensive collection of machine learning projects covering supervised learning, unsupervised learning, deep learning, and exploratory data analysis.

---

## 📁 Project Structure

```
Machine Learning/
├── README.md (main documentation)
│
├── Regression Analysis/
│   ├── Regression Analysis.ipynb
│   └── README.md
│
├── Supervised Learning - Classification/
│   ├── Supervised Learning - Classification.ipynb
│   └── README.md
│
├── Logistic Regression Classification/
│   ├── Logistic Regression Classification.ipynb
│   └── README.md
│
├── Classification Titanic Dataset/
│   ├── Classification Titanic Dataset.ipynb
│   └── README.md
│
├── Unsupervised Learning - Clustering/
│   ├── Unsupervised Learning - Clustering.ipynb
│   └── README.md
│
├── Neural Network - Perceptron Algorithm/
│   ├── Neural Network - Perceptron Algorithm.ipynb
│   └── README.md
│
├── VGG16 Image Classification/
│   ├── VGG16 Image Classification.ipynb
│   └── README.md
│
├── Image Preprocessing Trash Classification/
│   ├── Image Preprocessing Trash Classification.ipynb
│   └── README.md
│
├── Random Forest - Accident Prediction/
│   ├── Random Forest - Accident Prediction.ipynb
│   └── README.md
│
├── CNN Geometric Shapes/
│   ├── CNN Geometric Shapes.ipynb
│   └── README.md
│
└── EDA - COVID-19 Dataset Analysis/
    ├── EDA - COVID-19 Dataset Analysis.ipynb
    └── README.md
```

---

## 📋 Projects and Descriptions

### 1. **Linear Regression Analysis**
Folder: `Regression Analysis/`
- **Objective:** Build and evaluate linear regression models
- **Dataset:** USA Housing Dataset
- **Algorithm:** Linear Regression
- **Key Skills:** Data preprocessing, EDA, regression evaluation metrics
- **Output:** Property price prediction model
- **📖 Documentation:** See `Regression Analysis/README.md`

---

### 2. **Classification Pipeline**
Folder: `Supervised Learning - Classification/`
- **Objective:** Implement classification workflows
- **Dataset:** Titanic Dataset
- **Algorithms:** Multiple classification models
- **Key Skills:** Classification pipelines, feature selection, model comparison
- **Output:** Passenger survival prediction system
- **📖 Documentation:** See `Supervised Learning - Classification/README.md`

---

### 3. **Logistic Regression Model**
Folder: `Logistic Regression Classification/`
- **Objective:** Master logistic regression for binary classification
- **Dataset:** Titanic Dataset
- **Algorithm:** Logistic Regression
- **Key Skills:** Binary classification, confusion matrix, ROC-AUC analysis
- **Output:** Probability-based classification predictions
- **📖 Documentation:** See `Logistic Regression Classification/README.md`

---

### 4. **Advanced Classification**
Folder: `Classification Titanic Dataset/`
- **Objective:** Advanced classification techniques and optimization
- **Dataset:** Titanic Dataset
- **Algorithms:** Ensemble methods (Random Forest, Gradient Boosting)
- **Key Skills:** Feature engineering, ensemble learning, hyperparameter tuning
- **Output:** Optimized high-accuracy classifier
- **📖 Documentation:** See `Classification Titanic Dataset/README.md`

---

### 5. **Unsupervised Clustering**
Folder: `Unsupervised Learning - Clustering/`
- **Objective:** Discover patterns and group similar data
- **Dataset:** Biometric Dataset
- **Algorithm:** K-Means, Elbow Method, Silhouette Analysis
- **Key Skills:** Clustering, cluster optimization, cluster evaluation
- **Output:** Optimized clustering model with business insights
- **📖 Documentation:** See `Unsupervised Learning - Clustering/README.md`

---

### 6. **Neural Network Fundamentals**
Folder: `Neural Network - Perceptron Algorithm/`
- **Objective:** Understand foundational neural network concepts
- **Dataset:** Synthetic binary classification data
- **Algorithm:** Perceptron from scratch
- **Key Skills:** Neural network theory, activation functions, gradient descent
- **Output:** Working perceptron implementation with decision boundaries
- **📖 Documentation:** See `Neural Network - Perceptron Algorithm/README.md`

---

### 7. **Transfer Learning - VGG16**
Folder: `VGG16 Image Classification/`
- **Objective:** Leverage pre-trained models for image classification
- **Dataset:** Vehicle Classification Dataset (Kaggle)
- **Model:** VGG16 (ImageNet pre-trained)
- **Key Skills:** Transfer learning, fine-tuning, image augmentation
- **Output:** Production-ready vehicle classifier
- **📖 Documentation:** See `VGG16 Image Classification/README.md`

---

### 8. **Image Preprocessing & CNN**
Folder: `Image Preprocessing Trash Classification/`
- **Objective:** Build CNN with advanced preprocessing techniques
- **Dataset:** Waste Classification Dataset (Kaggle)
- **Model:** Custom CNN Architecture
- **Key Skills:** Image preprocessing, data augmentation, CNN training
- **Output:** Multi-class waste type classifier
- **📖 Documentation:** See `Image Preprocessing Trash Classification/README.md`

---

### 9. **Ensemble Methods - Random Forest**
Folder: `Random Forest - Accident Prediction/`
- **Objective:** Apply ensemble methods to real-world classification
- **Dataset:** Accident Severity Dataset
- **Model:** Random Forest Classifier
- **Key Skills:** Feature engineering, ensemble methods, evaluation metrics
- **Output:** Multi-class accident severity predictor
- **📖 Documentation:** See `Random Forest - Accident Prediction/README.md`

---

### 10. **CNN with Regularization**
Folder: `CNN Geometric Shapes/`
- **Objective:** Build robust CNN with regularization techniques
- **Dataset:** Geometric Shapes Dataset (Kaggle)
- **Model:** Custom CNN with Dropout & Early Stopping
- **Classes:** 3-class classification (Circle, Square, Triangle)
- **Key Skills:** CNN architecture, regularization, training callbacks
- **Features:** 
  - ✅ Dropout regularization
  - ✅ Early stopping mechanism
  - ✅ Complete ML pipeline
- **📖 Documentation:** See `CNN Geometric Shapes/README.md`

---

### 11. **Exploratory Data Analysis**
Folder: `EDA - COVID-19 Dataset Analysis/`
- **Objective:** Comprehensive data exploration and visualization
- **Dataset:** Synthetic COVID-19 Dataset
- **Focus:** Statistical analysis, pattern discovery, data quality
- **Key Skills:** EDA methodology, visualization, statistical testing
- **Output:** Data exploration report with actionable insights
- **📖 Documentation:** See `EDA - COVID-19 Dataset Analysis/README.md`

---

---

## 🛠️ Install Dependencies

### Requirements - Install All at Once
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras scipy statsmodels openpyxl pillow opencv-python kaggle
```

### Or Install by Category:

**Machine Learning & Data Analysis:**
```bash
pip install pandas numpy scikit-learn scipy statsmodels
```

**Visualization:**
```bash
pip install matplotlib seaborn
```

**Deep Learning (TensorFlow):**
```bash
pip install tensorflow keras
```

**Image Processing:**
```bash
pip install pillow opencv-python
```

## 🚀 Running the Notebooks

### Option 1: Google Colab (RECOMMENDED - Free GPU)

```python
# 1. Open https://colab.research.google.com

# 2. Upload notebook

# 3. For image datasets, mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

# 4. Setup Kaggle API (for datasets from Kaggle):
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 5. Download dataset:
!kaggle datasets download -d smeschke/four-shapes

# 6. Enable GPU: Runtime → Change runtime type → Select GPU
```

### Option 2: Local Machine

```bash
# 1. Install Jupyter
pip install jupyter notebook

# 2. Navigate to the folder
cd "c:/Users/acer/OneDrive/Gambar/KULIAH/KULIAH SEMESTER 6/Machine Learning"

# 3. Start Jupyter
jupyter notebook

# 4. Open browser at http://localhost:8888

# 5. Click the notebook file to run
```

### Option 3: VS Code

```bash
# 1. Install extension: Jupyter
# 2. Install extension: Python
# 3. Open folder
# 4. Open .ipynb file
# 5. Select Python kernel
# 6. Run cells
```

---

## � Directory Structure

```
Machine Learning/
│
├── README.md (dokumentasi utama)
│
├── Regression Analysis/
│   ├── Regression Analysis.ipynb
│   └── README.md
│
├── Supervised Learning - Classification/
│   ├── Supervised Learning - Classification.ipynb
│   └── README.md
│
├── Logistic Regression Classification/
│   ├── Logistic Regression Classification.ipynb
│   └── README.md
│
├── Classification Titanic Dataset/
│   ├── Classification Titanic Dataset.ipynb
│   └── README.md
│
├── Unsupervised Learning - Clustering/
│   ├── Unsupervised Learning - Clustering.ipynb
│   ├── README.md
│   └── berat_tinggi.csv (dataset)
│
├── Neural Network - Perceptron Algorithm/
│   ├── Neural Network - Perceptron Algorithm.ipynb
│   └── README.md
│
├── VGG16 Image Classification/
│   ├── VGG16 Image Classification.ipynb
│   ├── README.md
│   └── kendaraan/ (folder dengan images)
│       ├── train/
│       │   ├── class1/
│       │   └── class2/
│       └── val/
│           ├── class1/
│           └── class2/
│
├── Image Preprocessing Trash Classification/
│   ├── Image Preprocessing Trash Classification.ipynb
│   ├── README.md
│   └── TrashType_Image_Dataset/ (folder dengan images)
│
├── Random Forest - Accident Prediction/
│   ├── Random Forest - Accident Prediction.ipynb
│   ├── README.md
│   └── dataset_kecelakaan.csv (dataset)
│
├── CNN Geometric Shapes/
│   ├── CNN Geometric Shapes.ipynb
│   ├── README.md
│   └── geometric_shapes_dataset/ (folder dengan images)
│       ├── Circle/
│       ├── Square/
│       └── Triangle/
│
└── EDA - COVID-19 Dataset Analysis/
    ├── EDA - COVID-19 Dataset Analysis.ipynb
    ├── README.md
    └── synthetic_covid19_data.xlsx (dataset)
```

---

## 📚 Learning Path & Progression

### Phase 1: Foundations
- ✅ Linear Regression
- ✅ Classification Basics

### Phase 2: Intermediate Topics
- ✅ Advanced Classification
- ✅ Clustering & Unsupervised Learning

### Phase 3: Advanced Topics
- ✅ Neural Network Fundamentals (Perceptron)
- ✅ Deep Learning (Transfer Learning with VGG16)
- ✅ Image Processing (CNN from scratch)

### Phase 4: Specialized Topics
- ✅ Ensemble Methods (Random Forest)
- ✅ Advanced CNN (Regularization & Callbacks)

### Phase 5: Data Science Fundamentals
- ✅ EDA: Comprehensive Data Exploration

---

## 📄 License

Educational Use - Machine Learning Projects Collection

---

Happy Learning! 🚀📊🤖
