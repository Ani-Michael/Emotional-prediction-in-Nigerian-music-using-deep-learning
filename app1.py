import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Emotion Prediction App", layout="wide")

st.title("🎵 Nigerian Music Emotion Predictor")
st.markdown("Upload a CSV file of song features to predict their emotional category using an LSTM-based model.")

# Sidebar
with st.sidebar:
    st.header("🧠 About This App")
    st.markdown("""
    This app uses a deep learning model (LSTM) trained on Nigerian music features to predict emotional categories.
    Upload your dataset to visualize predictions, view emotion distributions, and download the results.
    """)

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
uploaded_file = st.file_uploader("📁 Upload a CSV file with audio features", type=["csv"])

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
        data_ordered = data[feature_columns]
        scaled = scaler.transform(data_ordered)
        reshaped = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))

        # Predict
        predictions = model.predict(reshaped)
        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        data["Predicted Emotion"] = predicted_labels

        # --- Visualization Section ---
        st.success("✅ Prediction complete!")

        st.subheader("🔍 Data Preview with Predictions")
        st.dataframe(data.head())

        st.subheader("📊 Emotion Distribution")
        emotion_counts = data["Predicted Emotion"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("🎯 Prediction Confidence")
        for i, row in data.iterrows():
            fig2, ax2 = plt.subplots()
            sns.barplot(x=label_encoder.classes_, y=predictions[i], ax=ax2)
            ax2.set_title(f"Track {i+1} Prediction Confidence")
            ax2.set_ylim(0, 1)
            st.pyplot(fig2)
            if i == 2:  # Only show first 3 tracks for confidence to save space
                break

        # --- Download Section ---
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Data with Predictions", csv, "predicted_emotions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
