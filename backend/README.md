# Backend API - Fetal Abnormality Detection

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Copy Trained Model
After training the ML model, copy it to the backend:
```bash
# From project root
cp ml/models/global_model.h5 backend/models/
```

### 3. Run the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Root Endpoint
```bash
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```bash
GET /health
```
Check if the API is running.

### 3. Model Information
```bash
GET /model-info
```
Get information about the loaded ML model.

**Response:**
```json
{
  "status": "Model loaded",
  "model_path": "/path/to/model",
  "input_shape": "224x224x3",
  "classes": ["Normal", "Abnormal"],
  "total_parameters": 1234567
}
```

### 4. Predict Abnormality
```bash
POST /predict
Content-Type: multipart/form-data
```

Upload an ultrasound image and get prediction.

**Request:**
- `file`: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "prediction": "Normal",
  "confidence": 0.85,
  "abnormality_probability": 0.15,
  "normal_probability": 0.85,
  "message": "No abnormalities detected. Fetus appears normal."
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ultrasound.jpg"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("ultrasound.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── ml_service.py        # ML model service
│   ├── db.py                # Database configuration
│   └── routes/
│       ├── __init__.py
│       ├── health.py        # Health check endpoint
│       └── prediction.py    # Prediction endpoints
├── models/
│   └── global_model.h5      # Trained ML model (copy from ml/)
├── .env                     # Environment variables
└── requirements.txt         # Python dependencies
```

## 🔒 Privacy & Security

### Privacy Features
- **No Data Storage**: Uploaded images are processed in-memory and not saved
- **Federated Learning**: Model trained without sharing patient data
- **HIPAA Compliant**: Designed for medical data privacy

### Security Recommendations for Production
1. **HTTPS**: Use SSL/TLS certificates
2. **Authentication**: Add JWT or OAuth2
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Strict file type checking
5. **CORS**: Restrict to specific frontend domains
6. **Logging**: Audit trail without storing images

## 🧪 Testing

### Test Health Endpoint
```bash
curl http://localhost:8000/health
```

### Test Model Info
```bash
curl http://localhost:8000/model-info
```

### Test Prediction
```bash
# Use a sample ultrasound image
curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_image.jpg"
```

## 🐛 Troubleshooting

### Error: "Model not found"
**Solution**: Copy the trained model from `ml/models/global_model.h5` to `backend/models/`

### Error: "Module not found"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Error: "Port already in use"
**Solution**: Change port: `uvicorn app.main:app --port 8001`

### Error: "CORS error from frontend"
**Solution**: Check CORS settings in `app/main.py`

## 📊 Performance

- **Inference Time**: ~100-200ms per image
- **Model Size**: ~10-50 MB (depending on architecture)
- **Memory Usage**: ~500 MB - 1 GB

## 🔄 Development

### Hot Reload
The `--reload` flag enables hot reload during development:
```bash
uvicorn app.main:app --reload
```

### Environment Variables
Create a `.env` file:
```env
MODEL_PATH=models/global_model.h5
LOG_LEVEL=info
```

## 🚀 Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment
- **AWS**: Elastic Beanstalk, ECS, or Lambda
- **Google Cloud**: Cloud Run or App Engine
- **Azure**: App Service or Container Instances

## 📝 Notes

- Model must be trained before running the backend
- Ensure sufficient memory for TensorFlow
- Use GPU for faster inference (optional)
- Monitor API performance in production
