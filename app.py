from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path(__file__).with_name("diabetesdata.csv")
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
INPUT_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age",
}
DEFAULT_VALUES = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


@st.cache_resource
def load_models() -> tuple[Pipeline, float]:
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURES]
    target = data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=20,
        stratify=target,
    )

    evaluation_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    evaluation_model.fit(x_train, y_train)
    test_accuracy = accuracy_score(y_test, evaluation_model.predict(x_test))

    prediction_model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    prediction_model.fit(features, target)
    return prediction_model, float(test_accuracy)


st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("Diabetes Prediction")
st.write(
    "Enter the health measurements below to see the prediction from the project's "
    "support vector machine model."
)
st.info(
    "This is an educational machine-learning demo, not a medical diagnosis or "
    "substitute for advice from a qualified healthcare professional."
)

model, test_accuracy = load_models()
st.metric("Held-out test accuracy", f"{test_accuracy:.1%}")

with st.form("prediction_form"):
    st.subheader("Health measurements")
    values: dict[str, float] = {}
    left, right = st.columns(2)
    for index, feature in enumerate(FEATURES):
        column = left if index % 2 == 0 else right
        step = 0.01 if feature in ("BMI", "DiabetesPedigreeFunction") else 1.0
        values[feature] = column.number_input(
            INPUT_LABELS[feature],
            min_value=0.0,
            value=float(DEFAULT_VALUES[feature]),
            step=step,
            format="%.2f" if step < 1 else "%.0f",
        )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    prediction = int(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    if prediction == 1:
        st.error("Model prediction: Positive (diabetic class)")
    else:
        st.success("Model prediction: Negative (non-diabetic class)")
    st.caption(
        "This result is a model classification, not a probability or a diagnosis. "
        "Please consult a healthcare professional about health concerns."
    )
