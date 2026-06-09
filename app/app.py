import streamlit as st
import joblib
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath("src"))
from recommendation_engine import get_recommendation

model = joblib.load("models/best_disease_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
symptom_columns = joblib.load("models/symptom_columns.pkl")

os.makedirs("results", exist_ok=True)

st.set_page_config(
    page_title="AI HealthCare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    padding: 0.8rem 1.4rem !important;
    max-width: 100% !important;
}

.stApp {
    background: #f4f7fb;
    font-family: "Segoe UI", sans-serif;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061b3a 0%, #08244d 70%, #031326 100%);
    width: 270px !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0.7rem !important;
    padding-bottom: 0.4rem !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.28rem !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] input {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 8px !important;
    height: 36px !important;
    min-height: 36px !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #ffffff !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #111827 !important;
}

section[data-testid="stSidebar"] hr {
    margin: 0.55rem 0 !important;
}

section[data-testid="stSidebar"] label {
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* MAIN TEXT */
.main-title {
    font-size: 26px;
    font-weight: 800;
    color: #071b3a;
    margin-bottom: 2px;
    line-height: 1.25;
}

.sub-title {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 12px;
}

/* CARDS */
.result-card {
    background: linear-gradient(90deg, #ecfdf5 0%, #ffffff 100%);
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 14px;
    margin-top: 10px;
    margin-bottom: 12px;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #071b3a;
    font-size: 22px;
    font-weight: 800;
}

.low {
    color: #15803d;
    font-size: 22px;
    font-weight: 800;
}

.medium {
    color: #ea580c;
    font-size: 22px;
    font-weight: 800;
}

.high {
    color: #dc2626;
    font-size: 22px;
    font-weight: 800;
}

.stButton button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 650 !important;
    height: 36px !important;
}

.stButton button:hover {
    background: #1d4ed8 !important;
    color: white !important;
}

.side-box {
    background: #ffffff;
    border: 1px solid #d9e2ef;
    border-radius: 13px;
    padding: 13px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    margin-bottom: 12px;
}

.side-title {
    color: #071b3a;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 6px;
}

.side-text {
    color: #334155;
    font-size: 13px;
    line-height: 1.6;
}

.status {
    background: #dcfce7;
    color: #15803d;
    padding: 7px 12px;
    border-radius: 8px;
    display: inline-block;
    font-weight: 700;
    margin-top: 7px;
}

.disclaimer {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #b91c1c;
    border-radius: 10px;
    padding: 10px;
    font-size: 13px;
}

.footer-text {
    font-size: 12px;
    color: #64748b;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏥 AI HealthCare")
    st.caption("Smart Disease Prediction")

    menu = st.radio(
        "Navigation",
        ["Dashboard", "Prediction History", "About System"]
    )

    st.divider()
    st.markdown("### 👤 Patient Information")

    patient_name = st.text_input("Patient Name", value="Ali Khan")
    age = st.text_input("Age", value="28")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.text_input("Height (cm)", value="175")
    weight = st.text_input("Weight (kg)", value="70")
    blood_pressure = st.text_input("Blood Pressure", value="120/80")

    if st.button("Update Information", use_container_width=True):
        st.success("Patient information updated.")

st.markdown(
    '<div class="main-title">AI-Based Disease Prediction & Healthcare Recommendation System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Professional symptom-based disease prediction and personalized healthcare recommendation dashboard.</div>',
    unsafe_allow_html=True
)

main_col, right_col = st.columns([4.4, 1.15], gap="medium")

with main_col:
    if menu == "Dashboard":
        with st.container(border=True):
            st.markdown("### 🔎 Select Symptoms")
            st.caption("Choose at least 3 symptoms for reliable prediction.")

            selected_symptoms = st.multiselect(
                "Symptoms",
                options=symptom_columns,
                placeholder="Search and select symptoms",
                label_visibility="collapsed"
            )

            b1, b2, b3 = st.columns([0.9, 0.9, 5])
            predict_btn = b1.button("Predict", use_container_width=True)
            clear_btn = b2.button("Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        if predict_btn:
            if len(selected_symptoms) < 3:
                st.warning("Please select at least 3 symptoms.")
            else:
                input_data = pd.DataFrame(0, index=[0], columns=symptom_columns)

                for symptom in selected_symptoms:
                    input_data[symptom] = 1

                prediction = model.predict(input_data)[0]
                disease = label_encoder.inverse_transform([prediction])[0]
                confidence = max(model.predict_proba(input_data)[0]) * 100
                rec = get_recommendation(disease)

                risk = rec["risk_level"].lower()
                risk_class = "low"
                if risk == "medium":
                    risk_class = "medium"
                elif risk == "high":
                    risk_class = "high"

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                r1, r2, r3 = st.columns([2.2, 1, 1])

                with r1:
                    st.markdown('<div class="metric-label">Predicted Disease</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">✅ {disease}</div>', unsafe_allow_html=True)

                with r2:
                    st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{confidence:.2f}%</div>', unsafe_allow_html=True)

                with r3:
                    st.markdown('<div class="metric-label">Risk Level</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="{risk_class}">{rec["risk_level"]}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### Healthcare Recommendation")

                c1, c2 = st.columns(2, gap="medium")

                with c1:
                    with st.container(border=True):
                        st.markdown("#### 📋 Disease Description")
                        st.write(rec["description"])

                    with st.container(border=True):
                        st.markdown("#### 🍽️ Diet Recommendation")
                        st.write(rec["diet"])

                with c2:
                    with st.container(border=True):
                        st.markdown("#### 🛡️ Precautions")
                        st.write(rec["precaution"])

                    with st.container(border=True):
                        st.markdown("#### 👨‍⚕️ Recommended Specialist")
                        st.info(rec["specialist"])

                st.markdown("""
                <div class="disclaimer">
                ⚠️ <b>Disclaimer:</b> This AI prediction is for educational decision-support only. 
                It is not a substitute for professional medical advice.
                </div>
                """, unsafe_allow_html=True)

                history = {
                    "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Patient_Name": patient_name,
                    "Age": age,
                    "Gender": gender,
                    "Height_cm": height,
                    "Weight_kg": weight,
                    "Blood_Pressure": blood_pressure,
                    "Selected_Symptoms": ", ".join(selected_symptoms),
                    "Predicted_Disease": disease,
                    "Confidence": round(confidence, 2),
                    "Risk_Level": rec["risk_level"],
                    "Specialist": rec["specialist"]
                }

                history_path = "results/prediction_history.csv"
                new_row = pd.DataFrame([history])

                if os.path.exists(history_path):
                    old = pd.read_csv(history_path)
                    new_row = pd.concat([old, new_row], ignore_index=True)

                new_row.to_csv(history_path, index=False)
                st.success("Prediction saved successfully.")

    elif menu == "Prediction History":
        st.markdown("### 📊 Prediction History")
        history_path = "results/prediction_history.csv"

        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            st.dataframe(history_df, use_container_width=True, height=430)

            st.download_button(
                "Download History CSV",
                history_df.to_csv(index=False).encode("utf-8"),
                "prediction_history.csv",
                "text/csv"
            )
        else:
            st.info("No prediction history found yet.")

    elif menu == "About System":
        st.markdown("### ℹ️ About System")
        with st.container(border=True):
            st.write("This AI system predicts possible diseases from selected symptoms using a trained SVM model.")
            st.write("It provides disease description, precautions, diet guidance, risk level, and specialist recommendation.")
            st.write("This project is designed for educational and academy-based ML demonstration.")

with right_col:
    st.markdown("""
    <div class="side-box">
        <div class="side-title">🧠 AI Model</div>
        <div class="side-text"><b>SVM Best Model</b></div>
        <div class="status">Active</div>
    </div>

    <div class="side-box">
        <div class="side-title">💡 Health Tips</div>
        <div class="side-text">
            ✅ Drink water<br>
            ✅ Balanced diet<br>
            ✅ Exercise regularly<br>
            ✅ Proper sleep<br>
            ✅ Health check-ups
        </div>
    </div>

    <div class="side-box">
        <div class="side-title">ℹ️ System Info</div>
        <div class="side-text">
            Symptom-based disease prediction system with personalized healthcare recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">© 2026 AI HealthCare System. Educational decision-support dashboard.</div>',
    unsafe_allow_html=True
)