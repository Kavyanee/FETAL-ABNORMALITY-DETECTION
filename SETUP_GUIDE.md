# 🚀 Complete Setup & Testing Guide

## 📋 Prerequisites Check

Before starting, ensure you have:

```bash
# Check Python version (need 3.10+)
python --version

# Check Node.js version (need 18+)
node --version

# Check npm
npm --version

# Check pip
pip --version
```

If any are missing:
- **Python**: Download from python.org
- **Node.js**: Download from nodejs.org
- **pip**: Usually comes with Python

---

## 🔧 Step-by-Step Setup

### Step 1: Project Setup

```bash
# Navigate to project directory
cd "d:\Finalyear Projects\Fetal-Abnormality-Detection"

# Verify structure
dir  # Windows
ls   # Linux/Mac
```

You should see: `ml/`, `backend/`, `frontend/`

---

### Step 2: ML Pipeline Setup

```bash
# Navigate to ML folder
cd ml

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# This will install:
# - TensorFlow (large, ~500MB)
# - Flower (Federated Learning)
# - OpenCV, NumPy, etc.
```

**Expected time**: 5-10 minutes (depending on internet speed)

**Troubleshooting**:
- **Flower version error**: Use `pip install -r requirements-minimal.txt` instead
- **TensorFlow fails**: Try `pip install tensorflow-cpu` (smaller, CPU-only)
- **Version conflicts**: Run `pip install --upgrade pip` first
- **If out of memory**: Close other applications
- **If specific package fails**: Install individually with `pip install package_name`

---

### Step 3: Train the Model

```bash
# Still in ml/ folder with venv activated
python train_federated.py
```

**What happens**:
1. Downloads ultrasound dataset from Kaggle
2. Preprocesses images (resize to 128x128, normalize)
3. Splits data across 3 hospitals
4. Trains for 2 federated rounds (configurable)
5. Saves model to `models/global_model.h5`

**Expected output**:
```
======================================================================
🏥 PRIVACY-PRESERVING FETAL ABNORMALITY DETECTION
   Federated Learning Training Pipeline
======================================================================

📊 Step 1: Data Preprocessing
----------------------------------------------------------------------
📥 Downloading dataset from Kaggle...
✅ Dataset downloaded to: /path/to/dataset
📂 Organizing dataset into 3 classes...
✅ Dataset organized:
   Normal: 500 images
   Benign: 500 images
   Malignant: 500 images

🏥 Creating federated data splits for 3 hospitals...
   Hospital 0: 500 images (Normal: 167, Benign: 167, Malignant: 166)
   Hospital 1: 500 images (Normal: 166, Benign: 167, Malignant: 167)
   Hospital 2: 500 images (Normal: 167, Benign: 166, Malignant: 167)
✅ Federated splits created

🌐 Step 2: Federated Learning Training
----------------------------------------------------------------------
🏥 Hospital 0 initialized:
   Training samples: 400
   Validation samples: 100

======================================================================
📍 FEDERATED ROUND 1/2
======================================================================
🏥 Hospital 0 training locally...
Epoch 1/2 ... loss: 0.8234 - accuracy: 0.6500
...

🔄 Aggregating weights from all hospitals...
📊 Evaluating global model...

======================================================================
✅ FEDERATED LEARNING COMPLETE
======================================================================

💾 Global model saved to: ml/models/global_model.h5
```

**Expected time**: 10-20 minutes (CPU), 3-5 minutes (GPU)

**Troubleshooting**:
- **Kaggle download fails**: Script will create synthetic data automatically
- **Out of memory**: Reduce `BATCH_SIZE` in `config.py`
- **Too slow**: Already optimized with FL_ROUNDS=2, LOCAL_EPOCHS=2

---

### Step 4: Evaluate the Model (Optional)

```bash
# Still in ml/ folder
python src/evaluation.py
```

**Expected output**:
```
======================================================================
🏥 FETAL ABNORMALITY DETECTION - MODEL EVALUATION
======================================================================
📥 Loading model from: ml/models/global_model.h5
✅ Model loaded successfully
📥 Loading test data...

📊 Test set size: 60 images
   Normal: 40
   Abnormal: 20

🔮 Generating predictions...

======================================================================
📊 MODEL EVALUATION METRICS
======================================================================

✅ Accuracy:  0.8167 (81.67%)
✅ Precision: 0.7500 (75.00%)
✅ Recall:    0.8000 (80.00%)
✅ F1-Score:  0.7742
✅ AUC-ROC:   0.8450

======================================================================

🏥 MEDICAL INTERPRETATION:
   • Out of 100 abnormal cases, we detect 80
   • Out of 100 predicted abnormal, 75 are truly abnormal

💾 Confusion matrix saved to: ml/models/confusion_matrix.png
💾 ROC curve saved to: ml/models/roc_curve.png

✅ Evaluation complete!
```

---

### Step 5: Backend Setup

```bash
# Open NEW terminal
cd "d:\Finalyear Projects\Fetal-Abnormality-Detection\backend"

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Expected time**: 2-5 minutes

---

### Step 6: Copy Model to Backend

```bash
# Still in backend/ folder

# Windows:
copy ..\ml\models\global_model.h5 models\

# Linux/Mac:
cp ../ml/models/global_model.h5 models/
```

**Verify**:
```bash
dir models\  # Windows
ls models/   # Linux/Mac
```

You should see: `global_model.h5`

---

### Step 7: Start Backend Server

```bash
# Still in backend/ folder
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
✅ Model loaded successfully from backend/models/global_model.h5
INFO:     Application startup complete.
```

**Test backend**:
Open browser: `http://localhost:8000`

You should see:
```json
{
  "message": "Fetal Abnormality Detection API",
  "version": "1.0.0",
  "endpoints": {
    "POST /predict": "Upload ultrasound image for prediction",
    "GET /model-info": "Get model information",
    "GET /health": "Health check"
  },
  "privacy": "This system uses Federated Learning - no patient data is shared",
  "documentation": "/docs"
}
```

**Test API docs**:
Open: `http://localhost:8000/docs`

**Troubleshooting**:
- **Port already in use**: Change port: `uvicorn app.main:app --port 8001`
- **Model not found**: Verify model copied to `backend/models/`
- **Import errors**: Reinstall dependencies

---

### Step 8: Frontend Setup

```bash
# Open NEW terminal (keep backend running)
cd "d:\Finalyear Projects\Fetal-Abnormality-Detection\frontend"

# Install dependencies
npm install
```

**Expected time**: 2-5 minutes

**Expected output**:
```
added 234 packages, and audited 235 packages in 45s

found 0 vulnerabilities
```

---

### Step 9: Start Frontend

```bash
# Still in frontend/ folder
npm run dev
```

**Expected output**:
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**Open browser**: `http://localhost:5173`

You should see the Fetal Abnormality Detection UI!

---

## 🧪 Testing the Complete System

### Test 1: Health Check

**Backend terminal**:
```bash
curl http://localhost:8000/health
```

**Expected**:
```json
{"status": "healthy"}
```

---

### Test 2: Model Info

**Browser**: `http://localhost:8000/model-info`

**Expected**:
```json
{
  "status": "Model loaded",
  "model_path": "backend/models/global_model.h5",
  "input_shape": "128x128x3",
  "classes": ["Normal", "Benign", "Malignant"],
  "total_parameters": 1234567
}
```

---

### Test 3: End-to-End Prediction

1. **Open frontend**: `http://localhost:5173`

2. **Prepare test image**:
   - Use any ultrasound image (JPEG/PNG)
   - Or use sample from `ml/data/processed/Normal/`, `Benign/`, or `Malignant/`

3. **Upload image**:
   - Click "Choose Image"
   - Select ultrasound image
   - See preview

4. **Analyze**:
   - Click "Analyze Image"
   - Wait 1-2 seconds

5. **Verify results**:
   - Prediction: "Normal", "Benign", or "Malignant"
   - Confidence: 0-100%
   - All 3 class probabilities displayed
   - Message shown

**Expected behavior**:
- Upload works smoothly
- Preview displays correctly
- Analysis takes < 2 seconds
- Results appear with animation
- No errors in console

---

### Test 4: Multiple Predictions

1. Click "Reset"
2. Upload different image from another class
3. Analyze again
4. Verify results update correctly

---

### Test 5: Error Handling

**Test A: Upload without backend**
1. Stop backend (Ctrl+C)
2. Try to analyze image
3. Should show error: "Failed to get prediction. Make sure the backend is running."

**Test B: Invalid file**
1. Try uploading non-image file (e.g., .txt)
2. Should be rejected by file input

---

## 📊 Verification Checklist

After setup, verify:

- [ ] ML model trained successfully
- [ ] Model file exists: `ml/models/global_model.h5`
- [ ] Model copied to: `backend/models/global_model.h5`
- [ ] Backend starts without errors
- [ ] Backend API docs accessible: `http://localhost:8000/docs`
- [ ] Frontend starts without errors
- [ ] Frontend UI loads: `http://localhost:5173`
- [ ] Can upload image
- [ ] Can get prediction for all 3 classes
- [ ] Results display correctly with all probabilities
- [ ] Can reset and try again

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt  # For Python
npm install                       # For Node.js
```

---

### Issue 2: Backend can't find model

**Solution**:
```bash
# Verify model exists
dir ml\models\global_model.h5  # Windows
ls ml/models/global_model.h5   # Linux/Mac

# Copy again
copy ml\models\global_model.h5 backend\models\  # Windows
cp ml/models/global_model.h5 backend/models/    # Linux/Mac
```

---

### Issue 3: Shape mismatch errors

**Cause**: Model expects 128x128 images, not 224x224

**Solution**: Already fixed in latest code. If error persists:
- Delete `ml/data/` folder
- Re-run `python train_federated.py`

---

### Issue 4: CORS errors in browser console

**Solution**:
Check `backend/app/main.py` has CORS middleware:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Issue 5: Port already in use

**Solution**:
```bash
# Backend - use different port
uvicorn app.main:app --port 8001

# Frontend - update backend URL in App.jsx
const response = await fetch('http://localhost:8001/predict', ...)
```

---

### Issue 6: Slow predictions

**Cause**: First prediction loads model (slow), subsequent ones are fast

**Solution**: This is normal. Wait for first prediction, then it's fast.

---

### Issue 7: Out of memory during training

**Solution**:
Edit `ml/config.py`:
```python
BATCH_SIZE = 8  # Reduce from 16
LOCAL_BATCH_SIZE = 4  # Reduce from 8
```

---

## 📱 Testing on Different Devices

### Desktop
- Works best
- Full features
- Fast performance

### Tablet
- Responsive layout
- Touch-friendly
- Good experience

### Mobile
- Stacked layout
- May need to zoom
- Slower upload

---

## 🎯 Performance Benchmarks

### Expected Performance

**Training**:
- CPU: 10-20 minutes
- GPU: 3-5 minutes

**Inference**:
- First prediction: 1-3 seconds (model loading)
- Subsequent: 100-300ms

**Memory Usage**:
- Backend: 500MB - 1GB
- Frontend: 50-100MB

---

## 📝 Final Checklist Before Demo

- [ ] All dependencies installed
- [ ] Model trained and copied
- [ ] Backend running and tested
- [ ] Frontend running and tested
- [ ] End-to-end flow works for all 3 classes
- [ ] Sample images prepared (Normal, Benign, Malignant)
- [ ] Backup plan ready (screenshots/video)
- [ ] Laptop charged
- [ ] Presentation slides ready

---

## 🎓 For Academic Submission

Include in your submission:
1. ✅ Source code (all folders)
2. ✅ Trained model (`global_model.h5`)
3. ✅ README.md (main documentation)
4. ✅ Demo video (optional but recommended)
5. ✅ Screenshots of results
6. ✅ Evaluation metrics (confusion matrix, ROC curve)
7. ✅ Report/Documentation

---

## 🚀 You're Ready!

If all tests pass, your system is ready for:
- ✅ Demo presentation
- ✅ Academic evaluation
- ✅ Viva defense
- ✅ Further development

**Good luck! 🎉**ferent image
3. Analyze again
4. Verify results update correctly

---

### Test 5: Error Handling

**Test A: Upload without backend**
1. Stop backend (Ctrl+C)
2. Try to analyze image
3. Should show error: "Failed to get prediction. Make sure the backend is running."

**Test B: Invalid file**
1. Try uploading non-image file (e.g., .txt)
2. Should be rejected by file input

---

## 📊 Verification Checklist

After setup, verify:

- [ ] ML model trained successfully
- [ ] Model file exists: `ml/models/global_model.h5`
- [ ] Model copied to: `backend/models/global_model.h5`
- [ ] Backend starts without errors
- [ ] Backend API docs accessible: `http://localhost:8000/docs`
- [ ] Frontend starts without errors
- [ ] Frontend UI loads: `http://localhost:5173`
- [ ] Can upload image
- [ ] Can get prediction
- [ ] Results display correctly
- [ ] Can reset and try again

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt  # For Python
npm install                       # For Node.js
```

---

### Issue 2: Backend can't find model

**Solution**:
```bash
# Verify model exists
dir ml\models\global_model.h5  # Windows
ls ml/models/global_model.h5   # Linux/Mac

# Copy again
copy ml\models\global_model.h5 backend\models\  # Windows
cp ml/models/global_model.h5 backend/models/    # Linux/Mac
```

---

### Issue 3: CORS errors in browser console

**Solution**:
Check `backend/app/main.py` has CORS middleware:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Issue 4: Port already in use

**Solution**:
```bash
# Backend - use different port
uvicorn app.main:app --port 8001

# Frontend - update backend URL in App.jsx
const response = await fetch('http://localhost:8001/predict', ...)
```

---

### Issue 5: Slow predictions

**Cause**: First prediction loads model (slow), subsequent ones are fast

**Solution**: This is normal. Wait for first prediction, then it's fast.

---

### Issue 6: Out of memory during training

**Solution**:
Edit `ml/config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
LOCAL_BATCH_SIZE = 8  # Reduce from 16
```

---

## 📱 Testing on Different Devices

### Desktop
- Works best
- Full features
- Fast performance

### Tablet
- Responsive layout
- Touch-friendly
- Good experience

### Mobile
- Stacked layout
- May need to zoom
- Slower upload

---

## 🎯 Performance Benchmarks

### Expected Performance

**Training**:
- CPU: 20-30 minutes
- GPU: 5-10 minutes

**Inference**:
- First prediction: 1-3 seconds (model loading)
- Subsequent: 100-300ms

**Memory Usage**:
- Backend: 500MB - 1GB
- Frontend: 50-100MB

---

## 📝 Final Checklist Before Demo

- [ ] All dependencies installed
- [ ] Model trained and copied
- [ ] Backend running and tested
- [ ] Frontend running and tested
- [ ] End-to-end flow works
- [ ] Sample images prepared
- [ ] Backup plan ready (screenshots/video)
- [ ] Laptop charged
- [ ] Presentation slides ready

---

## 🎓 For Academic Submission

Include in your submission:
1. ✅ Source code (all folders)
2. ✅ Trained model (`global_model.h5`)
3. ✅ README.md (main documentation)
4. ✅ Demo video (optional but recommended)
5. ✅ Screenshots of results
6. ✅ Evaluation metrics (confusion matrix, ROC curve)
7. ✅ Report/Documentation

---

## 🚀 You're Ready!

If all tests pass, your system is ready for:
- ✅ Demo presentation
- ✅ Academic evaluation
- ✅ Viva defense
- ✅ Further development

**Good luck! 🎉**
