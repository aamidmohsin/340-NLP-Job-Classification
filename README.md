# 340-NLP-Job-Classification

Job scams pose a significant threat to job seekers, leading to financial loss and identity theft. This research aims to develop a machine learning model to detect fraudulent job postings using the Employment Scam Aegean Dataset (EMSCAD), which contains 17,880 real-life job ads. We investigate the effectiveness of natural language processing (NLP) techniques and machine learning algorithms in distinguishing between legitimate and fraudulent job listings. By leveraging text-processing methods such as tokenization, feature extraction, and sentiment analysis, combined with classification models, we seek to identify linguistic markers and patterns indicative of scam job postings. The study provides insights into the key characteristics of fraudulent listings and evaluates the model's performance in detecting scams, contributing to the development of automated screening tools for online job platforms.

The full project report can be found in `report/Job_Classification_Report.pdf`

This report was developed in collaboration with Christian Kevin Sidharta & Marvin Roopchan.

## Project Directory Structure

```
340-NLP-Job-Classification/
  ├── data/
  │   ├── processed/
  │   │   └── dataset_cleaned.csv
  │   └── raw/
  │       └── DataSet.csv
  ├── docker/
  │   ├── app/
  │   │   └── main.py
  │   ├── model_assets/
  │   │   ├── *.joblib (trained models and vectorizers)
  │   ├── Dockerfile
  │   └── requirements.txt
  ├── frontend/
  │   ├── public/
  │   ├── src/
  │   └── ...
  ├── LICENSE
  ├── notebooks/
  │   ├── Job_Classification_EDA+Modelling.ipynb
  │   └── Job_Classification_Final_Training.ipynb
  ├── README.md
  └── report/
      └── Job_Classification_Report.pdf
```

## API Usage
This project includes a FastAPI-based REST API for making predictions on job descriptions. The API supports multiple trained models and returns confidence scores along with predictions.
