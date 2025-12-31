# Transfer Learning - VGG16 Image Classification

Leverage pre-trained deep learning models for efficient image classification using transfer learning techniques.

## 📋 Project Information

- **Notebook:** `VGG16 Image Classification.ipynb`
- **Topic:** Deep Learning - Transfer Learning
- **Dataset:** Vehicle Classification Dataset (Kaggle)
- **Model:** VGG16 (ImageNet pre-trained)
- **Complexity:** Intermediate to Advanced

## 🎯 Learning Objectives

In this project, you will learn:
- Konsep transfer learning
- Pre-trained CNN models
- Fine-tuning teknik
- Image preprocessing untuk deep learning
- Data augmentation
- Callbacks (EarlyStopping, ModelCheckpoint)
- Model evaluation untuk image classification

## 📊 Dataset

**Vehicle Classification Dataset** - Dataset gambar kendaraan untuk classification

- **Format:** Folder structure dengan subdirectories per class
- **Classes:** 2 (atau lebih) tipe kendaraan
- **Image Size:** Bervariasi (akan di-resize ke 224x224)
- **Type:** RGB images

### Folder Structure:
```
kendaraan/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
└── val/
    ├── class1/
    │   └── ...
    └── class2/
        └── ...
```

## 📥 Dataset Download dari Kaggle

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Setup API Key
# - Login ke https://www.kaggle.com/
# - Settings → Account → Create New API Token
# - Download kaggle.json
# - Letakkan di ~/.kaggle/ (Linux/Mac) atau C:\Users\[username]\.kaggle\ (Windows)
# - Beri permission: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)

# 3. Download Vehicle Dataset (example)
# Cari dataset di Kaggle yang sesuai, contohnya:
kaggle datasets download -d [dataset-id]

# 4. Extract
unzip [dataset-name].zip

# 5. Organize ke folder kendaraan/
```

**Popular Vehicle Datasets di Kaggle:**
- Vehicle Classification Dataset
- Car Brand Images
- Vehicle Type Classification
- Road Traffic Management Data

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn pillow
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **TensorFlow/Keras** | Deep learning framework |
| **numpy** | Numerical operations |
| **pandas** | Data manipulation |
| **matplotlib** | Visualization |
| **PIL/Pillow** | Image processing |
| **scikit-learn** | Metrics & utilities |

## 🚀 Cara Menjalankan

### Local Machine (GPU recommended)
```bash
# Install GPU support (optional but recommended)
pip install tensorflow[and-cuda]  # For NVIDIA GPUs

# Run notebook
jupyter notebook VGG16\ Image\ Classification.ipynb
```

### Google Colab (with GPU)
1. Upload notebook ke Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload atau mount drive dengan dataset
4. Run cells

## 📝 Inti Notebook

### 1. **Import & Setup**
- Import TensorFlow, Keras, utilities
- Set random seeds untuk reproducibility
- Configure GPU (jika available)

### 2. **Dataset Preparation**
- Define dataset path
- Set image size (224x224 untuk VGG16)
- Set batch size (32 typical)

### 3. **Data Loading dengan ImageDataGenerator**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # or 'categorical'
)
```

### 4. **Data Augmentation**
- Rotation, shifting, flipping
- Zoom, shearing
- Helps prevent overfitting

### 5. **Load Pre-trained VGG16 Model**
```python
from tensorflow.keras.applications import VGG16

# Load VGG16 tanpa top layers
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model weights
base_model.trainable = False
```

### 6. **Build Custom Model**
- Add custom layers on top VGG16:
  - Flatten layer
  - Dense layers
  - Dropout untuk regularization
  - Output layer (binary/categorical)

```python
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # or 'sigmoid' untuk binary
])
```

### 7. **Compile Model**
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # or 'categorical_crossentropy'
    metrics=['accuracy']
)
```

### 8. **Define Callbacks**
```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)
```

### 9. **Train Model**
```python
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)
```

### 10. **Evaluate & Visualize**
- Plot training history (loss, accuracy)
- Confusion matrix
- Classification report
- Sample predictions

### 11. **Fine-tuning (Optional)**
- Unfreeze beberapa layer terakhir base_model
- Train dengan learning rate lebih kecil
- Improve accuracy lebih lanjut

### 12. **Save Model**
```python
model.save('vehicle_classifier.h5')
# Atau save ke SavedModel format
model.save('vehicle_classifier')
```

## 📈 Output yang Dihasilkan

- **Training History Plots**
  - Train/val loss curve
  - Train/val accuracy curve
- **Confusion Matrix** - Prediction analysis
- **Classification Report** - Precision, recall, F1
- **Sample Predictions** - Test images dengan predictions
- **Model Weights** - Saved model file

## 💡 Key Concepts

### Transfer Learning
Menggunakan model yang sudah di-train pada dataset besar untuk solve related tasks:
- **Advantages:**
  - Faster training
  - Better performance dengan less data
  - Leverage learned features
- **Approaches:**
  - Feature extraction (freeze base, train only top)
  - Fine-tuning (unfreeze some layers, train with low LR)

### VGG16 Architecture
Convolutional neural network dengan:
- 13 convolutional layers
- 3 fully connected layers
- Total ~138M parameters
- Pre-trained pada ImageNet
- Input: 224×224×3 RGB images
- Output: 1000 classes (modify untuk custom task)

### Data Augmentation
Teknik untuk increase training data diversity:
- Rotation, shifting, zooming
- Flipping, shearing
- Brightness/contrast changes
- Helps prevent overfitting

### Dropout
Regularization technique yang randomly deactivate neurons:
- Prevents co-adaptation
- Forces network belajar robust features
- Typical: 0.3-0.5

### Early Stopping
Callback untuk stop training ketika validation loss tidak improve:
- Prevent overfitting
- Save computational resources
- Restore best model weights

## 📥 Kaggle Dataset Download Script

```bash
#!/bin/bash

# Setup Kaggle API
echo "Setting up Kaggle API..."
pip install kaggle

# Create .kaggle directory
mkdir -p ~/.kaggle

# Copy kaggle.json (assumed dari download sebelumnya)
# cp ~/Downloads/kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json

# Download dataset
echo "Downloading vehicle dataset..."
kaggle datasets download -d [dataset-id]

# Extract
echo "Extracting..."
unzip [dataset-name].zip

# Organize ke folder struktur yang benar
echo "Organizing dataset..."
# Custom script untuk organize files

echo "Done!"
```

**Alternative: Manual Download dari Kaggle Website**
1. Visit https://www.kaggle.com/datasets
2. Search for "Vehicle Classification" or "Car Dataset"
3. Click "Download"
4. Extract to the `kendaraan/` folder

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce batch_size, use smaller images, use Colab with GPU |
| Overfitting | Increase dropout, more augmentation, more data, early stopping |
| Poor accuracy | More epochs, adjust learning rate, check data quality, fine-tune |
| Slow training | Use GPU, reduce model size, reduce dataset size |
| File not found | Check path, ensure dataset in correct folder |

## 💻 Quick Reference Code

```python
# Complete pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
train_dir = 'kendaraan/train'
val_dir = 'kendaraan/val'
img_size = (224, 224)
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20).flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size
)

# Model
base = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base.trainable = False

model = Sequential([
    base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=50, 
          callbacks=[EarlyStopping(patience=10)])
```
