import pandas as pd

DISEASE_INFO_PATH = "data/disease_info.csv"

def get_recommendation(disease):
    df = pd.read_csv(DISEASE_INFO_PATH)

    row = df[df["Disease"].str.lower() == disease.lower()]

    if not row.empty:
        row = row.iloc[0]
        return {
            "description": row["Description"],
            "precaution": row["Precaution"],
            "diet": row["Diet"],
            "specialist": row["Specialist"],
            "risk_level": row["Risk_Level"]
        }

    return {
        "description": "Disease detected based on selected symptoms.",
        "precaution": "Consult a qualified doctor for proper diagnosis.",
        "diet": "Follow a balanced diet and stay hydrated.",
        "specialist": "General Physician",
        "risk_level": "Unknown"
    }