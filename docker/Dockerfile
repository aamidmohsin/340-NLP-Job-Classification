FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('omw-1.4', quiet=True)"

COPY app/ app/
COPY model_assets/ model_assets/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]