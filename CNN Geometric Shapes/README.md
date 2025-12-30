# Advanced CNN with Regularization Techniques

Build robust CNN models with dropout regularization and early stopping callbacks for shape classification.

## 📋 Project Information

- **Notebook:** `CNN Geometric Shapes.ipynb`
- **Topic:** Deep Learning - CNN with Regularization
- **Dataset:** Geometric Shapes Dataset (Kaggle)
- **Model:** Custom CNN with Dropout & Early Stopping
- **Classes:** 3-class classification (Circle, Square, Triangle)
- **Complexity:** Advanced

## 🎯 Learning Objectives

Pada ujian akhir ini, Anda akan:
- Build CNN model dari scratch
- Implement dropout untuk regularization
- Use early stopping untuk prevent overfitting
- Handle image datasets properly
- Model training & optimization
- Comprehensive evaluation
- Full deep learning pipeline

## 📊 Dataset

**Geometric Shapes Dataset** - Dataset gambar bentuk geometri

- **Classes:** Circle, Square, Triangle (3 classes)
- **Image Type:** RGB atau Grayscale
- **Image Size:** Varies (akan di-resize)
- **Source:** Kaggle

### Expected Folder Structure:
```
geometric_shapes_dataset/
├── Circle/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Square/
│   ├── image_001.jpg
│   └── ...
└── Triangle/
    ├── image_001.jpg
    └── ...
```

### After Processing:
```
dataset/
├── train/
│   ├── Circle/
│   ├── Square/
│   └── Triangle/
└── val/
    ├── Circle/
    ├── Square/
    └── Triangle/
```

## 📥 Dataset Download dari Kaggle

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Setup API Key
# - Visit https://www.kaggle.com/settings/account
# - Click "Create New API Token"
# - Save kaggle.json
# - Place di ~/.kaggle/ (Linux/Mac) atau C:\Users\[username]\.kaggle\ (Windows)

# 3. Download dataset
kaggle datasets download -d smeschke/four-shapes

# 4. Extract
unzip four-shapes.zip

# Alternative search untuk similar datasets
# - geometric shapes classification
# - simple shapes dataset
# - basic shapes recognition
```

**Alternative: Manual Download**
1. Go to https://www.kaggle.com/datasets
2. Search "geometric shapes" atau "four shapes"
3. Download dataset
4. Extract to working directory

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy matplotlib pillow scikit-learn opencv-python
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **TensorFlow/Keras** | Deep learning framework |
| **numpy** | Numerical operations |
| **matplotlib** | Visualization |
| **PIL/Pillow** | Image processing |
| **OpenCV** | Advanced image operations |
| **scikit-learn** | Metrics & evaluation |

## 🚀 Cara Menjalankan

### Google Colab (Recommended untuk GPU)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Download dataset
!kaggle datasets download -d smeschke/four-shapes
!unzip four-shapes.zip
```

### Local Machine
```bash
jupyter notebook CNN\ Geometric\ Shapes.ipynb
```

## 📝 Inti Ujian

### 1. **Setup & Import**
```python
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

### 2. **Dataset Exploration**
- List classes & counts
- Display sample images
- Check image properties (size, format, channels)

### 3. **Data Organization**
Organize dari raw dataset ke train/val structure:

```python
# Define paths
source_base = 'geometric_shapes_dataset'
target_base = 'dataset'
labels = ['Circle', 'Square', 'Triangle']

# Create directories
for label in labels:
    for split in ['train', 'val']:
        os.makedirs(f'{target_base}/{split}/{label}', exist_ok=True)

# Split & copy files (80-20 ratio)
for label in labels:
    files = os.listdir(f'{source_base}/{label}')
    np.random.shuffle(files)
    
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    # Copy training files
    for f in train_files:
        src = f'{source_base}/{label}/{f}'
        dst = f'{target_base}/train/{label}/{f}'
        shutil.copy(src, dst)
    
    # Copy validation files
    for f in val_files:
        src = f'{source_base}/{label}/{f}'
        dst = f'{target_base}/val/{label}/{f}'
        shutil.copy(src, dst)
```

### 4. **Image Preprocessing Setup**

```python
img_size = (128, 128)  # or (224, 224)
batch_size = 32

# Training data generator dengan augmentation
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

# Validation data generator (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    f'{target_base}/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    f'{target_base}/val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
```

### 5. **CNN Model Architecture**

**REQUIREMENT: Must include Dropout**

```python
num_classes = 3  # Circle, Square, Triangle

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # REQUIRED: Dropout
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # REQUIRED: Dropout
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # REQUIRED: Dropout
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # REQUIRED: Dropout
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # REQUIRED: Dropout
    layers.Dense(num_classes, activation='softmax')
])

model.summary()
```

### 6. **Compile Model**

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 7. **Callbacks Setup**

**REQUIREMENT: Must include Early Stopping**

```python
callbacks = [
    # REQUIRED: Early Stopping
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduce learning rate jika plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
```

### 8. **Model Training**

```python
epochs = 100  # With early stopping, akan stop lebih awal

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)
```

### 9. **Training History Visualization**

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid()

# Accuracy plot
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
```

### 10. **Model Evaluation**

```python
# Evaluate pada validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Get predictions
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_names = list(val_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_names))
```

### 11. **Sample Predictions Visualization**

```python
# Get batch dari validation set
images, labels = next(val_generator)

# Predict
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels, axis=1)

# Get class names
class_indices = val_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

# Plot
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.ravel()

for i in range(min(9, len(images))):
    axes[i].imshow(images[i])
    true_label = class_names[true_classes[i]]
    pred_label = class_names[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    
    color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                     color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### 12. **Model Saving**

```python
# Save final model
model.save('geometric_shapes_classifier.h5')

# Atau SavedModel format
model.save('geometric_shapes_classifier')

print("Model saved successfully!")
```

## 📈 Expected Output

- **Training curves** - Loss & accuracy over epochs
- **Confusion matrix** - Prediction analysis
- **Classification report** - Per-class metrics
- **Sample predictions** - Images dengan true/predicted labels
- **Saved model** - Trained weights
- **Summary statistics** - Overall performance

## 💡 Key Concepts

### Dropout
Regularization technique untuk prevent overfitting:
- Randomly deactivate neurons during training
- Typical rates: 0.2-0.5
- Forces network learn redundant features
- Effectively trains ensemble of networks

```
With Dropout(0.5):
50% of neurons deactivated randomly at each training step
```

### Early Stopping
Callback untuk stop training based on validation performance:
- Monitor validation loss atau accuracy
- Stop jika tidak improve untuk N epochs (patience)
- Restore best weights
- Prevent overfitting, save computation

**Parameters:**
- `monitor`: Metric to watch (val_loss, val_accuracy)
- `patience`: Epochs to wait sebelum stop (10-20 typical)
- `restore_best_weights`: Restore ke best epoch

### CNN Architecture
```
Input Image
    ↓
Conv Layers (extract features)
    ↓
Pooling Layers (reduce dimensions)
    ↓
Dropout (regularization)
    ↓
Flatten
    ↓
Dense Layers (classification)
    ↓
Output (softmax untuk multi-class)
```

### Class Distribution
Untuk 3 classes balanced:
- Circle: ~33%
- Square: ~33%
- Triangle: ~33%

### Evaluation Metrics untuk Multi-class
- **Accuracy:** Overall correctness
- **Precision per class:** TP/(TP+FP)
- **Recall per class:** TP/(TP+FN)
- **F1-Score:** Harmonic mean
- **Macro average:** Average across classes
- **Weighted average:** Account untuk class frequency

## ⚠️ Common Issues & Troubleshooting

| Problem | Solusi |
|---------|--------|
| Out of Memory | Reduce batch_size, smaller images, Colab GPU |
| Overfitting | Increase dropout, more augmentation, early stopping |
| Poor accuracy | Check data quality, adjust architecture, tune hyperparameters |
| Slow training | Use GPU, reduce model complexity |
| Early stopping tidak trigger | Check patience value, monitoring metric |
| File not found | Verify paths, check dataset structure |

## ✅ UAS Requirements Checklist

- [ ] Load geometric shapes dataset
- [ ] Organize train/val folders
- [ ] Create data generators dengan augmentation
- [ ] Build CNN model **WITH DROPOUT** ✓ (REQUIRED)
- [ ] Setup **EARLY STOPPING callback** ✓ (REQUIRED)
- [ ] Setup other callbacks (ModelCheckpoint, ReduceLROnPlateau)
- [ ] Train model
- [ ] Plot training history
- [ ] Evaluate model (confusion matrix, classification report)
- [ ] Visualize sample predictions
- [ ] Save model
- [ ] Write comprehensive report
- [ ] Document all steps dengan comments
- [ ] Submit .ipynb file

## 💻 Complete Pipeline Template

```python
# Full UAS Solution Template
import os, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup
source_base = 'geometric_shapes_dataset'
target_base = 'dataset'
img_size = (128, 128)
batch_size = 32
num_classes = 3

# 2. Organize dataset
# [copy-paste code dari section 3 di atas]

# 3. Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(f'{target_base}/train', 
                                              target_size=img_size, batch_size=batch_size)
val_gen = val_datagen.flow_from_directory(f'{target_base}/val', 
                                          target_size=img_size, batch_size=batch_size)

# 4. Build model WITH DROPOUT
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),  # DROPOUT
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),  # DROPOUT
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # DROPOUT
    layers.Dense(num_classes, activation='softmax')
])

# 5. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Callbacks WITH EARLY STOPPING
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),  # EARLY STOPPING
    ModelCheckpoint('best.h5', monitor='val_accuracy', save_best_only=True)
]

# 7. Train
history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks)

# 8. Evaluate & Visualize
# [copy-paste dari sections 9-11 di atas]

# 9. Save
model.save('geometric_shapes_classifier.h5')
```

## 📚 Referensi

- [CNN Architecture - TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)
- [Dropout Paper](https://arxiv.org/abs/1207.0580)
- [Early Stopping - Keras](https://keras.io/api/callbacks/early_stopping/)
- [Geometric Shapes Dataset - Kaggle](https://www.kaggle.com/datasets/smeschke/four-shapes)

## ✅ Grading Criteria

| Kriteria | Bobot |
|----------|-------|
| Dataset Organization | 10% |
| Model Architecture (with Dropout) | 20% |
| Early Stopping Implementation | 15% |
| Training & Evaluation | 20% |
| Visualization & Analysis | 15% |
| Code Quality & Documentation | 10% |
| Report & Insights | 10% |

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ UAS Submission

## 📝 Final Notes

✓ **MUST INCLUDE:**
- Dropout layers di model
- Early Stopping callback
- Full pipeline dari data loading to evaluation
- Proper documentation dengan comments
- Visualizations & analysis
- Saved model file

Good luck with your UAS!
