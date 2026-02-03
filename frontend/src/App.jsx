import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setPrediction(null)
      
      // Create preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  // Handle prediction
  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      
      // Transform prediction to show Normal Fetus or Abnormal Fetus
      let displayPrediction = data.prediction
      let displayMessage = data.message
      let combinedConfidence = data.confidence
      
      if (data.prediction === 'Benign' || data.prediction === 'Malignant') {
        displayPrediction = 'Abnormal Fetus'
        // Combine probabilities for Benign and Malignant
        combinedConfidence = data.probabilities.Benign + data.probabilities.Malignant
        displayMessage = 'Abnormal condition detected. Immediate specialist consultation required.'
      } else if (data.prediction === 'Normal') {
        displayPrediction = 'Normal Fetus'
      }
      
      // Transform probabilities to show only Normal and Cancerous Fetus (Benign + Malignant)
      const transformedData = {
        ...data,
        prediction: displayPrediction,
        confidence: combinedConfidence,
        message: displayMessage,
        probabilities: {
          'Normal': data.probabilities.Normal,
          'Cancerous Fetus': data.probabilities.Benign + data.probabilities.Malignant
        }
      }
      
      setPrediction(transformedData)
    } catch (err) {
      setError('Failed to get prediction. Make sure the backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Reset form
  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🏥 Fetal Abnormality Detection</h1>
        <p className="subtitle">Privacy-Preserving AI System using Federated Learning</p>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="upload-card">
            <h2>Upload Ultrasound Image</h2>
            
            <div className="file-input-wrapper">
              <input
                type="file"
                id="file-input"
                accept="image/*"
                onChange={handleFileSelect}
                className="file-input"
              />
              <label htmlFor="file-input" className="file-label">
                {selectedFile ? selectedFile.name : 'Choose Image'}
              </label>
            </div>

            {preview && (
              <div className="preview-section">
                <h3>Preview</h3>
                <img src={preview} alt="Preview" className="preview-image" />
              </div>
            )}

            <div className="button-group">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                className="btn btn-primary"
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
              
              {(selectedFile || prediction) && (
                <button onClick={handleReset} className="btn btn-secondary">
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>
        </div>

        {prediction && (
          <div className="results-section">
            <div className="results-card">
              <h2>Analysis Results</h2>
              
              <div className={`prediction-badge ${prediction.prediction.toLowerCase().replace(/\s+/g, '-')}`}>
                {prediction.prediction}
              </div>

              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {Object.entries(prediction.probabilities).map(([className, prob]) => (
                  <div className="metric" key={className}>
                    <span className="metric-label">{className} Probability</span>
                    <span className="metric-value">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              <div className="message">
                <p>{prediction.message}</p>
              </div>

              {prediction.prediction === 'Abnormal Fetus' && (
                <div className="warning">
                  ⚠️ This is an AI-assisted diagnosis. Please consult with a medical professional for confirmation.
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <div className="privacy-badge">
          🔒 Privacy-Preserving: No patient data is stored or shared
        </div>
        <p>Powered by Federated Learning | TensorFlow | FastAPI | React</p>
      </footer>
    </div>
  )
}

export default App
