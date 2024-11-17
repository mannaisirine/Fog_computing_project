from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)

# Configuration de l'application
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Chemin vers le modèle de prédiction
MODEL_PATH = 'C:/Users/USER/Desktop/fog/client2/models/model_weights.h5'
model = load_model(MODEL_PATH)

# Charger les labels des émotions
emotion_labels = ['happy', 'sad', 'angry', 'fear', 'neutral', 'disgust', 'pleasant surprise']
enc = OneHotEncoder()
enc.fit(pd.DataFrame(emotion_labels))

# Page d'accueil avec formulaire de téléversement
@app.route('/')
def index():
    return render_template('index.html')

# Route pour le téléversement de fichier et la prédiction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier sélectionné", 400

    file = request.files['file']
    if file.filename == '':
        return "Nom de fichier vide", 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Prétraitement et prédiction
        emotion = detect_emotion(filepath)

        # Redirection vers la page de résultat avec l'émotion prédite
        return redirect(url_for('result', emotion=emotion))

# Route pour afficher le résultat de la prédiction
@app.route('/result/<emotion>')
def result(emotion):
    return render_template('result.html', emotion=emotion)

# Prétraitement du fichier audio pour le modèle
def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None, duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

# Fonction pour détecter l'émotion à partir du fichier audio
def detect_emotion(audio_file):
    processed_audio = preprocess_audio(audio_file)
    predictions = model.predict(processed_audio)
    
    # Afficher les valeurs brutes de la prédiction
    print(f"Valeurs de prédiction (brutes) : {predictions}")
    
    # Inverser la transformation OneHot et obtenir l'étiquette prédite
    predicted_label = enc.inverse_transform(predictions)
    emotion = predicted_label[0][0]
    print(f"Émotion prédite : {emotion}")
    
    return emotion

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
