# Brain Tumor Detection Project

This project implements a complete machine learning pipeline for detecting brain tumors from MRI images. It includes both classification and regression tasks:

- **Image Classification:** Identifies whether an MRI image contains a tumor.
- **Object Detection (Regression):** Predicts bounding box coordinates for tumors in positive cases.

---

## Dataset Pre-processing

### Image-Label Matching

- Images without corresponding YOLO-format label files were discarded.
- Only entries with class `1` (indicating presence of tumor) were used in the regression model.
- For classification, tumor status was inferred from the presence of class `1` labels.

### Classification Pre-processing

**Approach 1: Image Duplication**

- Images with multiple tumors (multiple lines in the label file) were duplicated.
- Each copy was associated with a single bounding box.
- This increased dataset size but ignored multi-tumor complexity.

**Approach 2: Structured CSV Format**

- A `.csv` file was generated with each row containing:
  - Image filename
  - Tumor status (0 for negative, 1 for positive)
  - A column with:
    - Empty list `[]` for negative images
    - A 1D array for one bounding box
    - A 2D array for multiple bounding boxes
- This approach allowed for training on images with multiple tumors and was adopted in the final pipeline.

### Regression Pre-processing

- Only positive tumor images were used.
- Multi-line labels were preserved.
- Data was split into training and validation sets before model training.

---

## Image Classification Models

### Classical Models

Several traditional machine learning models were tested initially:

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.73     |
| Decision Tree       | 0.59     |
| Random Forest       | 0.80     |
| XGBoost             | 0.78     |
| Logistic NN         | 0.72     |
| CNN (from scratch)  | 0.51     |

Random Forest outperformed other classical models but lacked spatial context, motivating the use of deep learning.

### Deep Learning Models

**Custom CNN**

- Used Conv2D, MaxPooling, and Dropout layers.
- Accuracy was limited due to the small dataset and no pretrained features.

**Transfer Learning**

Pretrained models were fine-tuned for binary classification:

- ResNet18 and ResNet20
- ResNet50
- MobileNetV3
- EfficientNet

All models used:

- A new classification head
- EarlyStopping and ReduceLROnPlateau callbacks
- RandomizedSearch for hyperparameter optimization

**Data Augmentation**

Included random flips, zoom, brightness/contrast, and rotations to improve generalization and reduce overfitting.

---

## Tumor Coordinate Prediction (Regression)

### YOLOv5

Initial object detection trials used YOLOv5. The data was pre-processed into YOLOv5 format using only positive tumor labels. Results were decent but not competitive with YOLOv8.

### YOLOv8 (Final Implementation)

**Training Setup**

- Used `yolov8s` pretrained weights
- Only tumor-positive images were included
- Class `1` labels were relabeled to class `0` (binary problem)
- Augmentations: RandAugment, MixUp, color jittering, flipping, zooming, shearing
- Loss function parameters and optimizer (AdamW) were fine-tuned

**Training Configuration**

```python
epochs = 200
imgsz = 512
batch = 16
lr0 = 0.002
dropout = 0.2
optimizer = 'AdamW'
cos_lr = True
label_smoothing = 0.01
auto_augment = 'randaugment'
```

**Performance**

| Metric            | Value     |
|-------------------|-----------|
| Mean Precision    | 0.779     |
| Mean Recall       | 0.765     |
| F1 Score          | 0.772     |
| mAP@0.5           | 0.825     |
| mAP@0.5:0.95      | 0.592     |

**Model Path**

```
enhanced_training_runs/final_boosted_trial_v2/weights/best.pt
```

---

## Usage

### Classification

Use the fine-tuned transfer learning model (e.g., EfficientNet or ResNet) for binary tumor detection:

```python
model.predict(image_path)  # Returns 0 or 1
```

### YOLOv8 Prediction

Use the trained YOLOv8 model to predict tumor bounding boxes in MRI images:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(source='path_to_images/', save=True, conf=0.25)
```

---

## Export

Both classification and detection models can be exported and downloaded:

```python
from google.colab import files
files.download('best.pt')
files.download('final_metrics2.csv')
```

---

## Summary

This project delivers a dual-model pipeline for MRI-based brain tumor detection. It includes robust data cleaning, careful model selection, and both classical and deep learning techniques. The object detection module provides strong localization capabilities, while the classifier offers efficient tumor presence prediction. The pipeline is designed to support future clinical integration and research expansion.
