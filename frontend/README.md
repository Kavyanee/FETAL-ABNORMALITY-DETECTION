# Frontend - Fetal Abnormality Detection

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run Development Server
```bash
npm run dev
```

The app will be available at: `http://localhost:5173`

### 3. Build for Production
```bash
npm run build
```

## 🎨 Features

- **Image Upload**: Drag-and-drop or click to upload ultrasound images
- **Real-time Preview**: See uploaded image before analysis
- **AI Prediction**: Get instant Normal/Abnormal classification
- **Confidence Scores**: View detailed probability metrics
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Medical UI**: Clean, professional interface for healthcare

## 🔌 Backend Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`

Make sure the backend is running before using the frontend:
```bash
cd backend
uvicorn app.main:app --reload
```

## 📱 User Flow

1. **Upload Image**: Click "Choose Image" or drag-and-drop
2. **Preview**: Review the uploaded ultrasound image
3. **Analyze**: Click "Analyze Image" button
4. **Results**: View prediction with confidence scores
5. **Reset**: Clear and upload a new image

## 🎨 UI Components

### Header
- Title and subtitle
- Privacy-preserving messaging

### Upload Section
- File input with custom styling
- Image preview
- Action buttons (Analyze, Reset)
- Error handling

### Results Section
- Prediction badge (Normal/Abnormal)
- Confidence metrics
- Probability scores
- Medical interpretation message
- Warning for abnormal cases

### Footer
- Privacy badge
- Technology stack info

## 🔧 Configuration

### API Endpoint
To change the backend URL, edit `App.jsx`:
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
})
```

### Styling
Customize colors and layout in `App.css`

## 🧪 Testing

### Manual Testing
1. Start backend: `cd backend && uvicorn app.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Upload a test ultrasound image
4. Verify prediction appears correctly

### Test Cases
- ✅ Upload valid image (JPEG, PNG)
- ✅ Upload without backend running (error handling)
- ✅ Upload non-image file (validation)
- ✅ Reset after prediction
- ✅ Multiple consecutive predictions

## 📦 Dependencies

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "vite": "^5.0.0"
}
```

No additional libraries needed - pure React!

## 🎯 Key Features Explained

### Privacy-First Design
- No image storage on server
- In-memory processing only
- Clear privacy messaging to users

### Error Handling
- Backend connection errors
- Invalid file types
- Loading states
- User-friendly error messages

### Responsive Layout
- Desktop: Side-by-side upload and results
- Mobile: Stacked layout
- Touch-friendly buttons

## 🚀 Deployment

### Vercel (Recommended)
```bash
npm run build
vercel deploy
```

### Netlify
```bash
npm run build
netlify deploy --prod --dir=dist
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "3000"]
```

## 🔒 Security Considerations

- **HTTPS**: Use SSL in production
- **CORS**: Configure backend to allow only your frontend domain
- **Input Validation**: File type and size checks
- **Rate Limiting**: Prevent abuse (implement on backend)

## 📝 Customization

### Change Colors
Edit `App.css` gradient:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Add Logo
Replace header text with image:
```jsx
<img src="/logo.png" alt="Hospital Logo" />
```

### Add More Metrics
Extend the metrics section in `App.jsx`:
```jsx
<div className="metric">
  <span className="metric-label">Your Metric</span>
  <span className="metric-value">{value}</span>
</div>
```

## 🐛 Troubleshooting

### Issue: "Failed to get prediction"
**Solution**: Ensure backend is running on port 8000

### Issue: CORS error
**Solution**: Check CORS settings in backend `main.py`

### Issue: Image not displaying
**Solution**: Check file format (JPEG, PNG supported)

### Issue: Slow predictions
**Solution**: Backend may be loading model for first time (normal)

## 📊 Performance

- **Initial Load**: ~1-2 seconds
- **Image Upload**: Instant
- **Prediction**: 100-500ms (depends on backend)
- **Bundle Size**: ~150KB (gzipped)

## 🎓 For Demo/Presentation

Key points to highlight:
1. **Clean UI**: Professional medical interface
2. **Real-time**: Instant predictions
3. **Privacy**: No data storage messaging
4. **Responsive**: Works on all devices
5. **User-friendly**: Simple 3-step process

## 📞 Next Steps

After setup:
1. Test with sample ultrasound images
2. Customize branding and colors
3. Add authentication (if needed)
4. Deploy to production
5. Monitor usage and performance
