# Quick Start Guide - ML Pipeline

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2. Configure Kaggle API (Optional)
If you want to use real ultrasound data:
```bash
# Create Kaggle account and get API token
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)
```

### 3. Train the Model
```bash
# Federated Learning (Privacy-Preserving)
python train_federated.py

# OR Centralized (Baseline for comparison)
python train_centralized.py
```

### 4. Evaluate the Model
```bash
python src/evaluation.py
```

## 📊 What Each Script Does

### `train_federated.py`
- Downloads ultrasound dataset
- Splits data across 3 simulated hospitals
- Trains model using Federated Learning
- Saves global model to `models/global_model.h5`
- **Privacy**: No raw images shared between hospitals

### `train_centralized.py`
- Pools all data together (baseline)
- Trains model centrally
- Saves model to `models/centralized_model.h5`
- **Privacy**: Violates privacy (for comparison only)

### `src/evaluation.py`
- Evaluates model performance
- Generates confusion matrix
- Plots ROC curve
- Calculates accuracy, precision, recall, F1-score

## 🏗️ Project Structure
```
ml/
├── config.py                    # All configuration parameters
├── train_federated.py           # Main FL training script
├── train_centralized.py         # Baseline training script
├── requirements.txt             # Python dependencies
├── src/
│   ├── data_preprocessing.py    # Dataset handling
│   ├── cnn_model.py            # CNN architecture
│   ├── federated_learning.py   # FL implementation
│   └── evaluation.py           # Model evaluation
├── data/                        # Dataset (auto-created)
│   ├── processed/              # Normal/Abnormal images
│   └── federated/              # Hospital-wise splits
└── models/                      # Saved models (auto-created)
    └── global_model.h5         # Final FL model
```

## 🎯 Expected Output

After training, you should see:
- `models/global_model.h5` - Trained model
- `models/confusion_matrix.png` - Confusion matrix visualization
- `models/roc_curve.png` - ROC curve
- Console output with accuracy, precision, recall metrics

## 🔧 Troubleshooting

### Issue: Kaggle dataset download fails
**Solution**: The script will automatically create synthetic data for demo

### Issue: Out of memory
**Solution**: Reduce `BATCH_SIZE` in `config.py`

### Issue: Training too slow
**Solution**: Reduce `FL_ROUNDS` or `LOCAL_EPOCHS` in `config.py`

### Issue: Low accuracy
**Solution**: 
- Increase `FL_ROUNDS` (more training)
- Enable transfer learning: Set `USE_TRANSFER_LEARNING = True` in `config.py`
- Increase dataset size

## 📝 Configuration

Edit `config.py` to customize:
- `NUM_CLIENTS`: Number of hospitals (default: 3)
- `FL_ROUNDS`: Federated learning rounds (default: 10)
- `LOCAL_EPOCHS`: Epochs per hospital per round (default: 5)
- `IMG_HEIGHT`, `IMG_WIDTH`: Image size (default: 224x224)
- `USE_TRANSFER_LEARNING`: Use MobileNetV2 (default: False)

## 🎓 For Academic Presentation

Key points to explain:
1. **Problem**: Medical data privacy in AI
2. **Solution**: Federated Learning
3. **How it works**: Local training + weight aggregation
4. **Privacy guarantee**: No raw data sharing
5. **Performance**: Similar to centralized training
6. **Real-world impact**: HIPAA/GDPR compliant

## 🔒 Privacy Explanation

**Traditional ML**:
```
Hospital A → [Patient Images] → Central Server ❌
Hospital B → [Patient Images] → Central Server ❌
Hospital C → [Patient Images] → Central Server ❌
```

**Federated Learning**:
```
Hospital A → [Model Weights Only] → Central Server ✅
Hospital B → [Model Weights Only] → Central Server ✅
Hospital C → [Model Weights Only] → Central Server ✅
```

## 📞 Next Steps

After successful training:
1. Copy `models/global_model.h5` to `backend/models/`
2. Update backend API to load this model
3. Test with frontend UI
4. Deploy for demo
