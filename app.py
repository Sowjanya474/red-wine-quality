import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Red Wine Quality Predictor", page_icon="🍷")

@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")

bundle = load_bundle()
model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

st.title("🍷 Red Wine Quality Predictor")
st.write(
    "Enter the chemical properties of a red wine to estimate whether it "
    "would be rated **good** (quality 7 or above)."
)

# name: (min, max, typical default, step)
ranges = {
    "fixed acidity": (4.6, 15.9, 7.9, 0.1),
    "volatile acidity": (0.12, 1.58, 0.52, 0.01),
    "citric acid": (0.0, 1.0, 0.26, 0.01),
    "residual sugar": (0.9, 15.5, 2.2, 0.1),
    "chlorides": (0.012, 0.611, 0.079, 0.001),
    "free sulfur dioxide": (1.0, 72.0, 14.0, 1.0),
    "total sulfur dioxide": (6.0, 289.0, 38.0, 1.0),
    "density": (0.99007, 1.00369, 0.9968, 0.0001),
    "pH": (2.74, 4.01, 3.31, 0.01),
    "sulphates": (0.33, 2.0, 0.62, 0.01),
    "alcohol": (8.4, 14.9, 10.2, 0.1),
}

values = {}
col1, col2 = st.columns(2)
for i, name in enumerate(features):
    lo, hi, default, step = ranges[name]
    with (col1 if i % 2 == 0 else col2):
        values[name] = st.number_input(
            name.title(), min_value=float(lo), max_value=float(hi),
            value=float(default), step=float(step),
            format="%.4f" if step < 0.01 else "%.2f",
        )

if st.button("Predict quality"):
    row = pd.DataFrame([values])[features]
    prob = model.predict_proba(row)[0, 1]
    st.progress(float(prob))
    st.write(f"Estimated probability of a good wine: **{prob:.1%}**")
    if prob >= threshold:
        st.success("Prediction: GOOD wine (quality 7 or above)")
    else:
        st.warning("Prediction: NOT in the good category")
    st.caption(
        f"Decision threshold: {threshold:.2f}. Trained on the UCI Wine Quality "
        "(red) dataset. For educational use only."
    )