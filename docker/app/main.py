# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import re
import os
from typing import Optional, Dict, Tuple, Any, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# punkt_tab is required for newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass  # punkt_tab may not be available in all NLTK versions

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# omw-1.4 is often needed with wordnet for better compatibility
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    try:
        nltk.download('omw-1.4', quiet=True)
    except Exception:
        pass  # omw-1.4 may not be available in all NLTK versions

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ASSETS_DIR = BASE_DIR / 'model_assets'

# cache for loaded models and vectorizers
model_cache: Dict[str, Any] = {}
vectorizer_cache: Dict[str, Any] = {}

# initialize text preprocessing components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def punctuation_remover(strings: List[str]) -> List[str]:
    """Remove punctuation from a list of strings."""
    cleaned_strings = []
    for string in strings:
        cleaned_string = re.sub(r'[^\w\s]', '', string)
        cleaned_strings.append(cleaned_string)
    return cleaned_strings

def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing extra whitespace
    4. Tokenizing
    5. Lemmatizing
    6. Removing stop words
    """
    # Handle None or empty strings
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = punctuation_remover([text])[0]
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(tokens)

# model configuration mapping
MODEL_CONFIG = {
    "nb_bow": ("nb_model_bow.joblib", "bow_vectorizer.joblib"),
    "nb_bow_res": ("nb_model_bow_res.joblib", "bow_vectorizer.joblib"),
    "nb_tfidf": ("nb_model_tfidf.joblib", "tfidf_vectorizer.joblib"),
    "nb_tfidf_res": ("nb_model_tfidf_res.joblib", "tfidf_vectorizer.joblib"),
    "lr_bow": ("lr_model_bow.joblib", "bow_vectorizer.joblib"),
    "lr_bow_res": ("lr_model_bow_res.joblib", "bow_vectorizer.joblib"),
    "lr_tfidf": ("lr_model_tfidf.joblib", "tfidf_vectorizer.joblib"),
    "lr_tfidf_res": ("lr_model_tfidf_res.joblib", "tfidf_vectorizer.joblib"),
}

def load_model_and_vectorizer(model_label: str) -> Tuple[Any, Any]:
    """
    Load model and vectorizer for the given label.
    Uses caching to avoid reloading the same model.
    """
    if model_label not in MODEL_CONFIG:
        raise ValueError(f"Unknown model label: {model_label}. Available models: {list(MODEL_CONFIG.keys())}")
    
    # check cache first
    if model_label in model_cache and model_label in vectorizer_cache:
        return model_cache[model_label], vectorizer_cache[model_label]
    
    # load from disk
    model_filename, vectorizer_filename = MODEL_CONFIG[model_label]
    model_path = MODEL_ASSETS_DIR / model_filename
    vectorizer_path = MODEL_ASSETS_DIR / vectorizer_filename
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Cache the loaded model and vectorizer
        model_cache[model_label] = model
        vectorizer_cache[model_label] = vectorizer
        
        print(f"Loaded model '{model_label}' successfully.")
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model assets not found for '{model_label}': {e}")

# initialize FastAPI and define input schema
app = FastAPI(title="Job Fraud Classifier API")

# Configure CORS to allow requests from frontend
# Get allowed origins from environment variable or use defaults for development
allowed_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Can be configured via CORS_ORIGINS env var
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictionRequest(BaseModel):
    job_description: str
    model_label: Optional[str] = "nb_bow"  # Default to nb_bow 

# prediction endpoint 
@app.post("/predict")
def predict_fraud(data: PredictionRequest):
    try:
        # Load the specified model and vectorizer
        model, vectorizer = load_model_and_vectorizer(data.model_label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # preprocess and clean the input text
    cleaned_text = preprocess_text(data.job_description)
    
    # apply same feature extraction logic
    text_input = [cleaned_text]
    text_features = vectorizer.transform(text_input)

    # get prediction
    prediction_code = model.predict(text_features)[0]
    
    # get confidence scores (probabilities) for each class
    # predict_proba returns [[prob_class_0, prob_class_1]] for binary classification
    probabilities = model.predict_proba(text_features)[0]
    confidence_score = float(probabilities[int(prediction_code)])
    
    # interpret the result
    result = "Fraudulent (Fake)" if prediction_code == 1 else "Non-Fraudulent (Real)"
    
    return {
        "prediction_label": result,
        "prediction_code": int(prediction_code),
        "confidence_score": confidence_score,
        "probabilities": {
            "non_fraudulent": float(probabilities[0]),
            "fraudulent": float(probabilities[1])
        },
        "input_text_length": len(data.job_description),
        "cleaned_text_length": len(cleaned_text),
        "model_used": data.model_label
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Job Fraud Classifier API is running."}

"""
# local testing
if __name__ == '__main__':
    print("\n--- Running Local Test ---")
    print(f"Testing {len(MODEL_CONFIG)} models\n")
    
    # Test data
    test_data_real = PredictionRequest(
        job_description="We are seeking an experienced data scientist for a salaried position at our corporate headquarters in Chicago.",
        model_label="nb_bow"  # Will be overridden per model
    )
    test_data_fake = PredictionRequest(
        job_description="entry. data entry work customer service skill position home",
        model_label="nb_bow"  # Will be overridden per model
    )
    
    # Test each model
    for model_label in MODEL_CONFIG.keys():
        print(f"\n{'='*60}")
        print(f"Testing Model: {model_label}")
        print(f"{'='*60}")
        
        try:
            # Test with real job description
            test_data_real.model_label = model_label
            print("\n--- Testing Real Job Description ---")
            real_result = predict_fraud(test_data_real)
            print(f"  Prediction: {real_result['prediction_label']}")
            print(f"  Confidence: {real_result['confidence_score']:.4f} ({real_result['confidence_score']*100:.2f}%)")
            print(f"  Probabilities:")
            print(f"    - Non-Fraudulent: {real_result['probabilities']['non_fraudulent']:.4f} ({real_result['probabilities']['non_fraudulent']*100:.2f}%)")
            print(f"    - Fraudulent: {real_result['probabilities']['fraudulent']:.4f} ({real_result['probabilities']['fraudulent']*100:.2f}%)")
            
            # Test with fake job description
            test_data_fake.model_label = model_label
            print("\n--- Testing Fake Job Description ---")
            fake_result = predict_fraud(test_data_fake)
            print(f"  Prediction: {fake_result['prediction_label']}")
            print(f"  Confidence: {fake_result['confidence_score']:.4f} ({fake_result['confidence_score']*100:.2f}%)")
            print(f"  Probabilities:")
            print(f"    - Non-Fraudulent: {fake_result['probabilities']['non_fraudulent']:.4f} ({fake_result['probabilities']['non_fraudulent']*100:.2f}%)")
            print(f"    - Fraudulent: {fake_result['probabilities']['fraudulent']:.4f} ({fake_result['probabilities']['fraudulent']*100:.2f}%)")
            
        except Exception as e:
            print(f"ERROR: Failed to test model '{model_label}': {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("--- End Local Test ---")
    print(f"{'='*60}\n")
"""