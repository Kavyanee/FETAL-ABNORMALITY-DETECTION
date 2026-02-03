# ⚡ Quick Start - 3-Class Fetal Abnormality Detection

## 🎯 What's New

This system now classifies ultrasound images into **3 classes**:
- **Normal**: Healthy fetus
- **Benign**: Non-threatening condition
- **Malignant**: Serious condition requiring immediate attention

## 🚀 Quick Setup (3 Steps)

### 1. Train Model (10-20 min)
```bash
cd ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_federated.py
```

### 2. Start Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy ..\ml\models\global_model.h5 models\
uvicorn app.main:app --reload
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## 📊 Model Configuration

**Image Size**: 128x128 (optimized for speed)
**Classes**: 3 (Normal, Benign, Malignant)
**FL Rounds**: 2 (fast training)
**Hospitals**: 3 (simulated)

---

## 🧪 Testing

1. Upload image from `ml/data/processed/Normal/`, `Benign/`, or `Malignant/`
2. Click "Analyze Image"
3. See prediction with all 3 class probabilities

---

## ✅ Expected Results

**Training Output**:
```
Hospital 0: 500 images (Normal: 167, Benign: 167, Malignant: 166)
Hospital 1: 500 images (Normal: 166, Benign: 167, Malignant: 167)
Hospital 2: 500 images (Normal: 167, Benign: 166, Malignant: 167)

FEDERATED ROUND 1/2
...
FEDERATED LEARNING COMPLETE
Model saved to: ml/models/global_model.h5
```

**Prediction Output**:
```json
{
  "prediction": "Normal",
  "confidence": 0.95,
  "probabilities": {
    "Normal": 0.95,
    "Benign": 0.03,
    "Malignant": 0.02
  },
  "message": "No abnormalities detected. Fetus appears normal."
}
```

---

## 🐛 Common Issues

**Shape mismatch error**: Delete `ml/data/` and retrain
**Model not found**: Copy model to `backend/models/`
**CORS error**: Backend CORS already configured
**Slow training**: Already optimized (2 rounds, 128x128 images)

---

## 📚 Full Documentation

- **Setup Guide**: `SETUP_GUIDE.md`
- **Demo Guide**: `DEMO_GUIDE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

---

**Ready to go! 🎉**
