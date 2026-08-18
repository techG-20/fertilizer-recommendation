from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.recommender import FERTILIZERS, fertilizer_candidates, recommend_rule_based

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fertilizer_model.joblib"

st.set_page_config(page_title="Fertilizer Recommendation", page_icon="🌱", layout="wide")
st.title("🌱 Smart Fertilizer Recommendation System")
st.caption("ML-assisted decision support for soil and crop inputs")

with st.sidebar:
    st.header("Farm inputs")
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 28.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0)
    moisture = st.number_input("Soil moisture (%)", 0.0, 100.0, 45.0)
    soil_type = st.selectbox("Soil type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
    crop_type = st.selectbox(
        "Crop type",
        ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
         "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"],
    )
    nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 20.0)
    phosphorous = st.number_input("Phosphorus (P)", 0.0, 200.0, 15.0)
    potassium = st.number_input("Potassium (K)", 0.0, 200.0, 10.0)
    predict = st.button("Get recommendation", type="primary", use_container_width=True)

values = {
    "Temperature": temperature, "Humidity": humidity, "Moisture": moisture,
    "Soil Type": soil_type, "Crop Type": crop_type,
    "Nitrogen": nitrogen, "Phosphorous": phosphorous, "Potassium": potassium,
}

if predict:
    recommendation = recommend_rule_based(values)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended fertilizer", recommendation.fertilizer)
    c2.metric("N-P-K", "-".join(map(str, map(int, recommendation.composition))))
    c3.metric("Rule match", f"{recommendation.score * 100:.1f}%")

    st.subheader("Why this was selected")
    st.write(recommendation.explanation)

    if recommendation.deficiencies:
        st.info("Detected nutrient gaps: " + ", ".join(recommendation.deficiencies))
    else:
        st.warning("No nutrient deficiency was detected by the demo reference bands.")

    st.subheader("Top rule-based candidates")
    candidates = fertilizer_candidates(values, top_k=3)
    st.dataframe(pd.DataFrame([
        {
            "Fertilizer": c.fertilizer,
            "N-P-K": "-".join(map(str, map(int, c.composition))),
            "Match": f"{c.score * 100:.1f}%",
        }
        for c in candidates
    ]), hide_index=True, use_container_width=True)

    if MODEL_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)
        X = pd.DataFrame([values])[pipeline["features"]]
        X_t = pipeline["preprocessor"].transform(X)
        model = pipeline["model"]

        probabilities = model.predict_proba(X_t)[0]
        ranking = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]

        st.subheader("ML model predictions")
        st.dataframe(pd.DataFrame([
            {"Fertilizer": label, "Model probability": f"{prob * 100:.1f}%"}
            for label, prob in ranking
        ]), hide_index=True, use_container_width=True)

        st.caption(
            "The ML model is trained on the repository's synthetic demo dataset. "
            "Replace it with a properly validated real dataset before research or deployment."
        )
    else:
        st.warning("No trained ML model found. Run `python -m src.train`.")

st.divider()
st.subheader("Fertilizer knowledge base")
st.dataframe(pd.DataFrame([
    {"Fertilizer": name, "N": n, "P": p, "K": k}
    for name, (n, p, k) in FERTILIZERS.items()
]), hide_index=True, use_container_width=True)

st.caption(
    "Verify fertilizer choice and dosage against a soil test, crop stage, local recommendations "
    "and the exact commercial product label."
)
