# AI-Based Disease Prediction & Personalized Healthcare Recommendation System

## Overview
This project predicts diseases from patient symptoms using a trained Support Vector Machine (SVM) model and provides personalized healthcare recommendations including precautions, diet plans, risk levels, and specialist recommendations.

## Features
- Disease Prediction using Machine Learning
- Personalized Healthcare Recommendations
- Risk Level Assessment
- Confidence Score Calculation
- Prediction History Storage
- CSV Export Functionality
- Professional Streamlit Dashboard

## Technology Stack
- Python
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Project Structure

Disease_Prediction/
├── app/
│   └── app.py
├── data/
│   ├── Training.csv
│   ├── Testing.csv
│   └── disease_info.csv
├── models/
│   ├── best_disease_model.pkl
│   ├── label_encoder.pkl
│   └── symptom_columns.pkl
├── results/
│   └── prediction_history.csv
├── src/
│   ├── train_model.py
│   ├── cross_validation.py
│   ├── predict.py
│   └── recommendation_engine.py
├── requirements.txt
└── README.md

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train_model.py
```

## Cross Validation

```bash
python src/cross_validation.py
```

## Test Prediction

```bash
python src/predict.py
```

## Run Dashboard

```bash
streamlit run app/app.py
```

## Model Performance

| Model | Accuracy |
|---------|----------|
| Decision Tree | 97.62% |
| Random Forest | 97.62% |
| SVM | 100.00% |

## Workflow

Patient Information
→ Symptom Selection
→ Data Preprocessing
→ SVM Model
→ Disease Prediction
→ Confidence Score
→ Risk Assessment
→ Recommendation Engine
→ Dashboard
→ Prediction History

## Disclaimer

This project is for educational and research purposes only and should not replace professional medical advice.

## Author

Sana Cheema
Machine Learning Engineer
AIVONEX
