# Model Accuracy Improvements Summary

## Dataset Analysis
**Class Distribution (CLASS IMBALANCE DETECTED):**
- Normal: 750 images
- Benign: 776 images  
- **Malignant: 4,507 images** (5.8x larger than others!)

**Problem:** Model was biased towards predicting Malignant class due to severe imbalance.

---

## Improvements Implemented

### 1. ✅ **Class Weight Handling** (Addresses Malignant Bias)
- **File:** `src/federated_learning.py`
- **What:** Computed automatic class weights using sklearn's `compute_class_weight()`
- **How:** Inverse class frequency weighting:
  ```
  Class Weight = Total Samples / (Num Classes × Samples per Class)
  Normal weight   = ~5033 / (3 × 750)   ≈ 2.24
  Benign weight   = ~5033 / (3 × 776)   ≈ 2.16
  Malignant weight = ~5033 / (3 × 4507) ≈ 0.37
  ```
- **Impact:** Normal and Benign classes now weighted more heavily during loss calculation

### 2. ✅ **Stronger Regularization (Dropout)**
- **File:** `src/cnn_model.py`
- **Change:** Increased dropout from 0.25-0.3 → 0.5
- **Why:** Random neuron dropout prevents overfitting on small datasets
- **Impact:** Reduces gap between train acc (89%) and val acc (63-80%)

### 3. ✅ **L2 Regularization (Weight Penalty)**
- **File:** `config.py` + `src/cnn_model.py`
- **What:** Added `L2_REGULARIZATION = 0.0001` to all Conv2D and Dense layers
- **How:** Penalizes large weights → prevents complex, overfitted patterns
- **Code:** `kernel_regularizer=keras.regularizers.l2(0.0001)`
- **Impact:** Encourages simpler, more generalizable features

### 4. ✅ **Reduced Epochs**
- **File:** `config.py`
- **Change:** LOCAL_EPOCHS: 5 → 3
- **Change:** EPOCHS_CENTRALIZED: 50 → 30
- **Why:** Prevents training too long on small data → stops overfitting earlier
- **Impact:** Better validation accuracy, stops before model memorizes

### 5. ✅ **Early Stopping Callback**
- **File:** `src/federated_learning.py`
- **What:** Stop training if validation loss doesn't improve for 5 epochs
- **Config:** `EARLY_STOPPING_PATIENCE = 5`, `EARLY_STOPPING_ENABLED = True`
- **Impact:** Automatically stops overfitting, saves best weights

### 6. ✅ **Memory-Efficient Data Loading** (Already Fixed)
- **File:** `src/data_preprocessing.py`
- **What:** tf.data.Dataset with on-the-fly image loading
- **Impact:** Prevents OutOfMemory errors during federated rounds

---

## Expected Accuracy Improvements

### Before Improvements:
- Normal/Benign Recall: **LOW** (model ignores these classes due to imbalance)
- Malignant Recall: **HIGH** (model biased towards this class)
- Validation Accuracy: 63-80% (overfitting gap: 26%)
- **Expected Test Accuracy: 65-75%** (Malignant-biased)

### After Improvements:
- Normal/Benign Recall: **IMPROVED** ↑ (class weights prioritize these)
- Malignant Recall: **Slightly Lower** (balanced with other classes)
- Validation Accuracy: **More Stable** (early stopping prevents overfitting)
- **Expected Test Accuracy: 75-85%** (balanced across all classes)

---

## Configuration Changes Summary

```python
# config.py changes:
LOCAL_EPOCHS = 3  # Was 5
DROPOUT_RATE = 0.5  # Was 0.3
L2_REGULARIZATION = 0.0001  # NEW
EPOCHS_CENTRALIZED = 30  # Was 50
EARLY_STOPPING_PATIENCE = 5  # Was 10
EARLY_STOPPING_ENABLED = True  # NEW
USE_CLASS_WEIGHTS = True  # NEW
AUTOMATIC_CLASS_WEIGHTS = True  # NEW
```

---

## Files Modified
1. ✅ `config.py` - Configuration parameters
2. ✅ `src/cnn_model.py` - L2 regularization in layers
3. ✅ `src/federated_learning.py` - Class weights + Early stopping
4. ✅ `src/data_preprocessing.py` - Memory-efficient loading (already done)

---

## How to Run with Improvements

```bash
cd ml
python train_federated.py
```

The model will now:
1. ✅ Weight Malignant class down (reduce bias)
2. ✅ Weight Normal/Benign class up (better detection)
3. ✅ Use stronger regularization (prevent overfitting)
4. ✅ Stop early when validation plateaus (avoid memorization)
5. ✅ Load data efficiently without memory errors

---

## Recommended Next Steps (If More Data Available)

If you get more ultrasound images in the future:
1. **Collect more Normal/Benign samples** (currently 750/776 vs 4507 for Malignant)
2. **Use medical imaging pre-trained models** (ResNet50/InceptionV3 trained on medical data)
3. **Data augmentation** (already implemented in data pipeline)
4. **Ensemble methods** (combine multiple models for voting)
5. **Focal Loss** (alternative to class weights for severe imbalance)

