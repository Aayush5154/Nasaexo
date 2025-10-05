import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------- Load model and preprocessing ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

medians = joblib.load("medians.pkl")
le_dict = joblib.load("label_encoders.pkl")

st.title("Exoplanet Prediction 🚀")
st.write("Input the KOI features to predict if it is a confirmed exoplanet.")

# ---------- Feature names ----------
feature_names = model.get_booster().feature_names  # all features used in training

# ---------- Friendly display names ----------
feature_display_names = {
    "kepid": "KepID",
    "kepoi_name": "KOI Name",
    "kepler_name": "Kepler Name",
    "koi_disposition": "Exoplanet Archive Disposition",
    "koi_pdisposition": "Disposition Using Kepler Data",
    "koi_score": "Disposition Score",
    "koi_fpflag_nt": "Not Transit-Like False Positive Flag",
    "koi_fpflag_ss": "Stellar Eclipse False Positive Flag",
    "koi_fpflag_co": "Centroid Offset False Positive Flag",
    "koi_fpflag_ec": "Ephemeris Contamination False Positive Flag",
    "koi_period": "Orbital Period [days]",
    "koi_time0bk": "Transit Epoch [BKJD]",
    "koi_duration": "Transit Duration [hrs]",
    "koi_depth": "Transit Depth [ppm]",
    "koi_prad": "Planetary Radius [Earth radii]",
    "koi_teq": "Equilibrium Temperature [K]",
    "koi_insol": "Insolation Flux [Earth flux]",
    "koi_model_snr": "Transit Signal-to-Noise",
    "koi_steff": "Stellar Effective Temperature [K]",
    "koi_slogg": "Stellar Surface Gravity [log10(cm/s^2)]",
    "koi_srad": "Stellar Radius [Solar radii]",
    "ra": "RA [decimal degrees]",
    "dec": "Dec [decimal degrees]",
    "koi_kepmag": "Kepler-band [mag]"
}

# ---------- Collect user input ----------
user_input = {}
for feature in feature_names:
    display_name = feature_display_names.get(feature, feature)  # fallback to column name
    if feature in le_dict:
        user_input[feature] = st.text_input(f"{display_name} (categorical)")
    else:
        user_input[feature] = st.number_input(f"{display_name}", value=medians.get(feature, 0.0))

input_df = pd.DataFrame([user_input])

# ---------- Preprocessing ----------
for col in le_dict:
    if col in input_df.columns:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

for col in medians:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna(medians[col])

# ---------- Prediction ----------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"This KOI is predicted to be an EXOPLANET! 🌟 (Confidence: {proba:.2f})")
    else:
        st.warning(f"This KOI is predicted to be NOT an EXOPLANET. (Confidence: {1-proba:.2f})")
