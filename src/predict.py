import joblib
import pandas as pd
from recommendation_engine import get_recommendation

model = joblib.load("models/best_disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
symptom_columns = joblib.load("models/symptom_columns.pkl")


def predict_disease(selected_symptoms):
    input_data = pd.DataFrame(0, index=[0], columns=symptom_columns)

    for symptom in selected_symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    prediction = model.predict(input_data)[0]
    disease = label_encoder.inverse_transform([prediction])[0]

    recommendation = get_recommendation(disease)

    return disease, recommendation


if __name__ == "__main__":
    symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    disease, recommendation = predict_disease(symptoms)

    print("Predicted Disease:", disease)
    print("Recommendation:", recommendation)