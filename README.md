
[Uploading screen-capture (8).webm…]()

# AI-Based Disease Prediction & Personalized Healthcare Recommendation System
<img width="959" height="443" alt="p1" src="https://github.com/user-attachments/assets/5e8c5f93-e913-4d4f-9515-18fc42239539" />
<img width="960" height="403" alt="p2" src="https://github.com/user-attachments/assets/5d189ebb-0f7a-47b9-85af-dc734d0bf97b" />
<img width="959" height="437" alt="p3" src="https://github.com/user-attachments/assets/a8aac35f-b4cd-4ac1-8bac-3739201fc112" />
<img width="955" height="447" alt="p4" src="https://github.com/user-attachments/assets/0e1c2d49-0ab2-4561-9df9-422c8ead1033" />


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
