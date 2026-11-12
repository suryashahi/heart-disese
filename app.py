import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_complete_model.pkl")

st.title("❤️ Heart Disease Risk Prediction")
st.caption("Educational machine-learning demonstration — not a medical diagnosis.")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Place heart_disease_complete_model.pkl in the same folder as app.py.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# The app expects the saved object to be the complete fitted sklearn Pipeline.
if not hasattr(model, "named_steps") or not {"preprocessor", "classifier"}.issubset(model.named_steps):
    st.error(
        "The uploaded .pkl does not contain the complete prediction pipeline. "
        "It must include both 'preprocessor' and 'classifier' steps. "
        "A preprocessing-only object cannot make predictions."
    )
    st.stop()

classifier = model.named_steps["classifier"]
if not hasattr(classifier, "classes_"):
    st.error("The classifier in this .pkl is not fitted. Save the fitted pipeline after calling pipeline.fit(X_train, y_train).")
    st.stop()

st.subheader("Enter patient measurements")
st.info("Enter values from a clinical record. The categories below use the UCI Heart Disease dataset coding.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[(1, "Male (1)"), (0, "Female (0)")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest pain type", options=[1, 2, 3, 4], format_func=lambda x: {
            1: "1 — Typical angina", 2: "2 — Atypical angina", 3: "3 — Non-anginal pain", 4: "4 — Asymptomatic"
        }[x])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=50, max_value=700, value=200)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[(0, "No (0)"), (1, "Yes (1)")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: {
            0: "0 — Normal", 1: "1 — ST-T wave abnormality", 2: "2 — Left ventricular hypertrophy"
        }[x])
        thalach = st.number_input("Maximum heart rate achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise-induced angina", options=[(0, "No (0)"), (1, "Yes (1)")], format_func=lambda x: x[1])[0]
    with c3:
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", options=[1, 2, 3])
        ca = st.selectbox("Number of major vessels (0–3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[(3, "3 — Normal"), (6, "6 — Fixed defect"), (7, "7 — Reversible defect")], format_func=lambda x: x[1])[0]

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        st.divider()
        st.subheader("Prediction result")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(classifier.classes_)
            positive_index = classes.index(1) if 1 in classes else -1
            risk_score = float(probabilities[positive_index]) if positive_index >= 0 else None
        else:
            risk_score = None

        if prediction == 1:
            st.error("The model predicts the presence of heart disease.")
        else:
            st.success("The model predicts no heart disease.")

        if risk_score is not None:
            st.metric("Model-estimated probability for the disease class", f"{risk_score:.1%}")
            st.caption("This is a model probability, not a clinically validated personal risk estimate.")

        st.warning("This result is for educational purposes only. It cannot diagnose, rule out, or assess an individual's medical condition. Consult a qualified healthcare professional for medical decisions.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
