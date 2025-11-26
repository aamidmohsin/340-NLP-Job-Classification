# Job Fraud Classifier Frontend

A React.js frontend application for interacting with the Job Fraud Classifier API.

## Features

- Simple, intuitive interface for pasting job posting text
- Model selection dropdown to choose from different ML models
- Real-time prediction results with confidence scores
- Visual probability breakdown
- Responsive design for mobile and desktop

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- The backend API should be running (default: http://localhost:8080)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Configuration

**⚠️ IMPORTANT: API URL Configuration**

The API URL is stored in an environment variable for security. You **must** create a `.env` file before running the application.

1. Create a `.env` file in the `frontend` directory:
```bash
cd frontend
touch .env
```

2. Add your API URL to the `.env` file:
```
REACT_APP_API_URL=http://your-api-url:port
```

**Example:**
- For local development: `REACT_APP_API_URL=http://localhost:8080`
- For remote server: `REACT_APP_API_URL=http://your-server-ip:8080`

**Security Note:** The `.env` file is automatically ignored by git (via `.gitignore`), so your API URL will never be committed to the repository. Never commit sensitive URLs or credentials.

## Running the Application

Start the development server:
```bash
npm start
```

The app will open in your browser at `http://localhost:3000`.

## Building for Production

To create a production build:
```bash
npm run build
```

This creates an optimized build in the `build` folder that can be served by any static file server.

## Usage

1. Select a model from the dropdown (default: Naive Bayes - BOW)
2. Paste or type a job posting text in the textarea
3. Click "Analyze Job Posting" to get predictions
4. View the results including:
   - Prediction label (Fraudulent or Non-Fraudulent)
   - Confidence score
   - Probability breakdown for both classes
   - Model metadata

## Available Models

- **nb_bow**: Naive Bayes - BOW
- **nb_bow_res**: Naive Bayes - BOW (Resampled)
- **nb_tfidf**: Naive Bayes - TF-IDF
- **nb_tfidf_res**: Naive Bayes - TF-IDF (Resampled)
- **lr_bow**: Logistic Regression - BOW
- **lr_bow_res**: Logistic Regression - BOW (Resampled)
- **lr_tfidf**: Logistic Regression - TF-IDF
- **lr_tfidf_res**: Logistic Regression - TF-IDF (Resampled)

