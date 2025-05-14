import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Prediction App", layout="centered")

st.title("🎵 Nigerian Music Emotion Prediction")
st.write("Upload a song's audio features to predict the emotional category.")

# Load model and preprocessing tools
try:
    model = load_model("Final_Final_test.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f"🔴 Error loading model or preprocessing files: {e}")
    st.stop()

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with audio features", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        data = pd.read_csv(uploaded_file)

        # Check required columns
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Reorder and scale
        data = data[feature_columns]
        scaled = scaler.transform(data)
        reshaped = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))

        # Predict
        predictions = model.predict(reshaped)
        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

        # Show result
        data["Predicted Emotion"] = predicted_labels
        st.success("✅ Prediction complete!")
        st.write(data[["Predicted Emotion"]].head())

        # Option to download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_emotions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
