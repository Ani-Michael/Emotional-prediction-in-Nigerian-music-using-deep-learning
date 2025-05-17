# Emotional-prediction-in-Nigerian-music-using-deep-learning

## Overview
This project focuses on using a deep learning model to predict the emotional category in Nigerian songs based on their audio features. By training a model on labeled spotify data, the model classifies songs into five emotion-based categories:Happy, sad, angry, excited and calm. It aims to understand the emotional state of Nigerian music and support mood-based recommendations.


## Features

### 1. Data cleaning 
Data was gotten from an online database kaggle. Removed unnecessary columns that were not needed for processing(e.g song title, artist name.etc).

### 2. Data Preprocessing
Converted categorical emotion labels to numerical format using LabelEncoder. Then spiliting data into training, validation and test sets also applied scaling(StandardScaler) to normalize feature calues.

### 3. Feature Engineering
Used relevant audio features like danceablity, energy, tempo,etc. And also transformed data into 3D shape for LTSM imput: (samples,timesteps, features)

### 4.Model Building and Evaluation
Designed a deep learning model using LTSM(Long Short-Term Memory)layers. Used softmax output for multiclass classification and then tuned the model for 30 epochs to improve the accuracy. Measured accuracy and loss on training and validation sets, then evaluated the final model on a new testing dataset.

### 5. Deployment
Deployed the model using streamlit for interactive web use, loaded and saved models using joblib.

## Challenge faced
Input shape mismatch for LTSM layers.
Maintaining feature consistency between training and test data 
Handling Streamlit deployment issues like missing dependencies

## Why this Project matters
Enhances understanding of emotional patterns in Nigerian music.
Lays foundation for music recommendation engines based on mood
useful for platforms like spotify, boomplay and DJs creating mood playlists.
Provides bridges between machine learning in arts and entertainment.

## Next steps and Future Improvements
Add more audio-based features like MFCCs, Spectral contrast
Add CNN-LSTM architecture to learn both spatial and sequential audio features.
Build a recommendation engine that uses predicted emotions to suggest music.





## Live Demo
The emotion prediction model has been deployed using Streamlit, a lightweight Python framework for building interactive web applications.
Visit the hosted web app using the provided link: https://emotional-prediction-in-nigerian-music-using-deep-learning-ani.streamlit.app/
 Steps to Use the Demo
Open the Web App: Navigate to the provided Streamlit link.

Enter Feature Values:

Input audio features manually (or upload a .csv file depending on the interface).
Example features include: danceability, energy, liveness, tempo, valence.

Click ‘Predict’:
The model will process your input and return the emotion prediction in real time.

Note
The model only works with pre-processed features similar to those used during training.
All scaling and transformations used during training are automatically applied during inference.

