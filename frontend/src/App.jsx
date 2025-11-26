import React, { useState } from 'react';
import './App.css';

// API URL must be set via environment variable
// Create a .env file in the frontend directory with: REACT_APP_API_URL=http://your-api-url:port
const API_URL = process.env.REACT_APP_API_URL;

if (!API_URL) {
  console.error('REACT_APP_API_URL environment variable is not set. Please create a .env file.');
}

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [modelLabel, setModelLabel] = useState('nb_bow');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const modelOptions = [
    { value: 'nb_bow', label: 'Naive Bayes - BOW' },
    { value: 'nb_bow_res', label: 'Naive Bayes - BOW (Resampled)' },
    { value: 'nb_tfidf', label: 'Naive Bayes - TF-IDF' },
    { value: 'nb_tfidf_res', label: 'Naive Bayes - TF-IDF (Resampled)' },
    { value: 'lr_bow', label: 'Logistic Regression - BOW' },
    { value: 'lr_bow_res', label: 'Logistic Regression - BOW (Resampled)' },
    { value: 'lr_tfidf', label: 'Logistic Regression - TF-IDF' },
    { value: 'lr_tfidf_res', label: 'Logistic Regression - TF-IDF (Resampled)' },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!jobDescription.trim()) {
      setError('Please enter a job description');
      return;
    }

    // Client-side validation: check word count before sending request
    const wordCount = jobDescription.trim().split(/\s+/).filter(word => word.length > 0).length;
    if (wordCount < 10) {
      setResult({
        insufficient_data: true,
        message: 'Not enough information provided to determine whether posting is legitimate or fraudulent'
      });
      setError(null);
      return;
    }

    if (!API_URL) {
      setError('API URL is not configured. Please create a .env file with REACT_APP_API_URL set.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_description: jobDescription,
          model_label: modelLabel,
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to get prediction';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (parseError) {
          // If JSON parsing fails, use status text
          errorMessage = response.statusText || errorMessage;
        }
        
        // Check if error is due to insufficient data
        if (errorMessage.toLowerCase().includes('data input not long enough') || 
            errorMessage.toLowerCase().includes('not long enough')) {
          // Set as special result instead of error
          setResult({
            insufficient_data: true,
            message: 'not enough information to determine whether posting is legitimate or fraudulent'
          });
        } else {
          throw new Error(errorMessage);
        }
      } else {
        const data = await response.json();
        setResult(data);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while processing your request');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setJobDescription('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>Job Fraud Classifier</h1>
          <p className="subtitle">Detect fraudulent job postings using machine learning</p>
        </header>

        <form onSubmit={handleSubmit} className="form">
          <div className="form-group">
            <label htmlFor="model-select">Select Model:</label>
            <select
              id="model-select"
              value={modelLabel}
              onChange={(e) => setModelLabel(e.target.value)}
              className="model-select"
            >
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="job-description">Job Posting Text:</label>
            <textarea
              id="job-description"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job posting text here..."
              rows="10"
              className="textarea"
              disabled={loading}
            />
          </div>

          <div className="button-group">
            <button
              type="submit"
              className="submit-button"
              disabled={loading || !jobDescription.trim()}
            >
              {loading ? 'Analyzing...' : 'Analyze Job Posting'}
            </button>
            {(result || error) && (
              <button
                type="button"
                onClick={handleReset}
                className="reset-button"
              >
                Clear & Reset
              </button>
            )}
          </div>
        </form>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-container">
            <h2>Prediction Results</h2>
            
            {result.insufficient_data ? (
              <div className="insufficient-data-message">
                <p>{result.message}</p>
              </div>
            ) : (
              <>
                <div className={`prediction-badge ${result.prediction_code === 1 ? 'fraudulent' : 'legitimate'}`}>
                  <span className="prediction-label">{result.prediction_label}</span>
                  <span className="confidence-score">
                    {(result.confidence_score * 100).toFixed(2)}% confidence
                  </span>
                </div>

                <div className="metadata">
                  <div className="metadata-item">
                    <strong>Model Used:</strong> {result.model_used}
                  </div>
                  <div className="metadata-item">
                    <strong>Input Text Length:</strong> {result.input_text_length} characters
                  </div>
                  <div className="metadata-item">
                    <strong>Cleaned Text Length:</strong> {result.cleaned_text_length} characters
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

