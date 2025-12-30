# Image Preprocessing & CNN for Classification

Build robust CNN models with advanced image preprocessing and data augmentation techniques.

## 📋 Project Information

- **Notebook:** `Image Preprocessing Trash Classification.ipynb`
- **Topic:** Deep Learning - Image Preprocessing & CNN
- **Dataset:** Waste Classification Dataset (Kaggle)
- **Model:** Custom CNN Architecture
- **Complexity:** Intermediate to Advanced

## 🎯 Learning Objectives

In this project, you will learn:
- Teknik image preprocessing
- Data augmentation untuk increase training diversity
- Dataset preparation dari raw images
- Building CNN models dari scratch
- Image data pipeline
- Model training untuk image classification
- Handling imbalanced datasets

## 📊 Dataset

**Trash Type Image Dataset** - Dataset gambar sampah dengan berbagai kategori

- **Format:** Gambar dalam folder per kategori/kelas
- **Classes:** Multiple trash categories (e.g., organic, plastic, metal, paper, etc.)
- **Image Type:** RGB color images, various sizes
- **Total Images:** Tergantung dataset

### Expected Folder Structure:
```
TrashType_Image_Dataset/
├── category1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── category2/
│   ├── image1.jpg
│   └── ...
└── category3/
    └── ...
```

**After Processing:**
```
sorted_dataset/
├── train/
│   ├── category1/
│   ├── category2/
│   └── ...
└── val/
    ├── category1/
    ├── category2/
    └── ...
```

## 📥 Dataset Download dari Kaggle

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Setup Kaggle API Key
# - Go to https://www.kaggle.com/settings/account
# - Click "Create New API Token"
# - Download kaggle.json
# - Place di ~/.kaggle/ (Linux/Mac) atau C:\Users\[username]\.kaggle\ (Windows)

# Linux/Mac:
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell):
# Copy kaggle.json to C:\Users\[username]\.kaggle\

# 3. Search dan download trash dataset
kaggle datasets download -d asdasdasasdas/garbage-classification
# Atau cari dataset lain yang sesuai

# 4. Extract
unzip garbage-classification.zip

# 5. Organize ke folder struktur yang benar
# Notebook akan handle ini secara otomatis
```

**Popular Trash Datasets di Kaggle:**
- [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [TrashNet Dataset](https://www.kaggle.com/datasets/fedesoriano/trashnet)
- [Waste Classification Data](https://www.kaggle.com/datasets/mostafaabdelrahman/garbage-classification)

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy matplotlib pillow scikit-learn opencv-python shutil
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **TensorFlow/Keras** | Deep learning framework |
| **numpy** | Numerical operations |
| **matplotlib** | Visualization |
| **PIL/Pillow** | Image manipulation |
| **OpenCV (cv2)** | Advanced image processing |
| **scikit-learn** | Metrics & utilities |
| **shutil** | File operations |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook Image\ Preprocessing\ Trash\ Classification.ipynb
```

### Google Colab (Recommended)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload dataset atau download dari Kaggle
!kaggle datasets download -d asdasdasasdas/garbage-classification
```

## 📝 Inti Notebook

### 1. **Setup & Import**
```python
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
```

### 2. **Dataset Exploration**
- List semua kategori/classes
- Count images per category
- Display sample images
- Check image sizes & formats

### 3. **Image Preprocessing**
- Load images dari disk
- Resize ke uniform size (e.g., 128x128, 224x224)
- Normalisasi pixel values (0-1 atau -1 to 1)
- Check untuk corrupted images
- Handle imbalanced classes (optional)

```python
def load_and_preprocess_image(path, img_size=128):
    img = Image.open(path)
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0  # Normalize
    return img_array
```

### 4. **Dataset Splitting**
- Split menjadi train/val (80-20 atau 70-30)
- Stratified split untuk balanced distribution
- Shuffle data

```python
# Create train/val directories
for category in os.listdir(original_dataset_dir):
    src = os.path.join(original_dataset_dir, category)
    train_dst = os.path.join(train_dir, category)
    val_dst = os.path.join(val_dir, category)
    
    # Copy files dengan split ratio
    files = os.listdir(src)
    train_files = files[:int(0.8*len(files))]
    val_files = files[int(0.8*len(files)):]
    
    for f in train_files:
        shutil.copy(os.path.join(src, f), train_dst)
```

### 5. **Data Augmentation Setup**

**Train Generator dengan Augmentation:**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Val Generator (no augmentation, only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
```

### 6. **CNN Model Architecture**

Contoh sederhana:
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 7. **Visualisasi Augmented Images**
- Display original vs augmented images
- Show transformation effects

### 8. **Model Training**
```python
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
)
```

### 9. **Training History Visualization**
- Plot training/validation loss
- Plot training/validation accuracy
- Analyze for overfitting

### 10. **Model Evaluation**
```python
# Test pada validation set
val_loss, val_accuracy = model.evaluate(val_generator)

# Predictions
y_pred = model.predict(val_generator)

# Confusion matrix & classification report
from sklearn.metrics import confusion_matrix, classification_report
y_true = val_generator.classes
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
print(classification_report(y_true, y_pred_classes, 
                          target_names=val_generator.class_indices))
```

### 11. **Visualisasi Predictions**
- Display test images dengan true/predicted labels
- Highlight correct/incorrect predictions
- Show confidence scores

### 12. **Model Saving**
```python
model.save('trash_classifier.h5')
# Atau
model.save('trash_classifier')  # SavedModel format
```

## 📈 Output yang Dihasilkan

- **Dataset Statistics** - Class distribution, image count
- **Sample Visualizations** - Original & augmented images
- **Training Curves** - Loss & accuracy plots
- **Confusion Matrix** - Prediction analysis
- **Classification Report** - Precision, recall, F1 per class
- **Sample Predictions** - Test images dengan predictions
- **Trained Model** - Saved weights

## 💡 Key Concepts

### Image Preprocessing
Prepare raw images untuk deep learning:
- **Resizing:** Uniform input size untuk network
- **Normalization:** Scale pixel values (0-1 atau -1 to 1)
- **Format conversion:** RGB, grayscale, etc.

### Data Augmentation
Teknik untuk virtually increase dataset size:
- **Rotation:** Random rotations
- **Shifting:** Width/height shifts
- **Zooming:** Random zooms
- **Flipping:** Horizontal/vertical flips
- **Shearing:** Shear transformations

**Benefits:**
- Prevent overfitting
- Improve generalization
- Leverage limited data
- Simulate real-world variations

### CNN Architecture Components

**Convolutional Layer:**
- Extract local features
- Learnable filters/kernels
- Preserves spatial relationships

**Pooling Layer:**
- Reduce spatial dimensions
- Extract dominant features
- Max pooling common

**Fully Connected Layer:**
- Classification
- Feature aggregation

**Dropout:**
- Regularization
- Prevent overfitting
- Rate: typical 0.3-0.5

### Categorical vs Binary Classification
- **Binary:** 2 classes, sigmoid activation, binary_crossentropy
- **Categorical:** >2 classes, softmax activation, categorical_crossentropy

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce batch_size, image size, model size |
| Slow training | Use GPU, reduce image size, reduce epochs |
| Overfitting | More augmentation, increase dropout, add data |
| Poor accuracy | Check data quality, adjust hyperparameters, larger model |
| File path errors | Check working directory, use absolute paths |
| Class imbalance | class_weight or resampling |

## 💻 Complete Example

```python
# Full pipeline example
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

# Paths
original_dir = 'TrashType_Image_Dataset'
base_dir = 'sorted_dataset'
img_size = 128
batch_size = 32

# Step 1: Create train/val directories
for split in ['train', 'val']:
    for category in os.listdir(original_dir):
        os.makedirs(f'{base_dir}/{split}/{category}', exist_ok=True)

# Step 2: Copy files dengan 80-20 split
for category in os.listdir(original_dir):
    files = os.listdir(f'{original_dir}/{category}')
    split_idx = int(0.8 * len(files))
    
    for f in files[:split_idx]:
        shutil.copy(f'{original_dir}/{category}/{f}', 
                   f'{base_dir}/train/{category}/{f}')
    for f in files[split_idx:]:
        shutil.copy(f'{original_dir}/{category}/{f}', 
                   f'{base_dir}/val/{category}/{f}')

# Step 3: Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=30, 
                               zoom_range=0.2).flow_from_directory(
    f'{base_dir}/train', target_size=(img_size, img_size), batch_size=batch_size)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    f'{base_dir}/val', target_size=(img_size, img_size), batch_size=batch_size)

# Step 4: Build model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train
model.fit(train_gen, validation_data=val_gen, epochs=30)

# Step 6: Save
model.save('trash_classifier.h5')
```

## 📚 Referensi

- [Image Preprocessing - TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation Guide](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [CNN Architecture Basics](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Kaggle Trash Datasets](https://www.kaggle.com/search?q=trash+classification)

## ✅ Checklist

- [ ] Download dataset dari Kaggle
- [ ] Explore dataset structure
- [ ] Create train/val folders
- [ ] Setup data generators
- [ ] Visualize augmented images
- [ ] Build CNN model
- [ ] Define callbacks
- [ ] Train model
- [ ] Plot training history
- [ ] Evaluate model
- [ ] Visualize predictions
- [ ] Save model

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete

## 🚀 Next Steps

- Experiment dengan different architectures (ResNet, EfficientNet)
- Implement transfer learning
- Fine-tune pre-trained models
- Deploy sebagai API
- Mobile optimization
