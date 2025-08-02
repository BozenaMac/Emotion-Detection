# Emotion-Detection

This project implements a Convolutional Neural Network (CNN) to classify facial emotions into 7 categories using TensorFlow and OpenCV.

---

## Features

- Face detection and preprocessing using OpenCV  
- Data augmentation for robust training  
- Handling class imbalance with computed class weights  
- Training of a deep CNN model with multiple convolutional blocks  
- Model evaluation with accuracy, classification report, and confusion matrix  
- Model saved as `emotion_model_final.h5`  
- **Streamlit web app for uploading face images**  
- **Automatic face detection in uploaded images using Haar cascades**  
- **Emotion classification on detected faces with confidence scores**  
- **Grad-CAM heatmaps visualizing areas influencing emotion prediction**  
- **Bar plots showing probabilities of all emotions**  
- **ChatGPT-powered empathetic text recommendations based on detected emotion**  
- **Humor feature: Jokes triggered when sadness is detected to improve mood**  

---

## Dataset

The model was trained on a dataset created by combining and preprocessing several publicly available datasets from Kaggle.

---

## Collaboration

This project was a joint effort by Paulina Stępniewska, Dominik Charasim, Adrian Biernacki, and me.

---

## Business Case

The business case for this project is to use it in conjunction with face ID technology. When unlocking a phone, the model reads the user's emotion, and ChatGPT provides an appropriate suggestion, for example, telling a joke if sadness is detected—to improve user's mood.

---

## Note

The TensorFlow app is included, but OpenAI ChatGPT prompt integration code (from `.toml`) is omitted.

