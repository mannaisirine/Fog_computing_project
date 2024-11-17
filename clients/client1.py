import requests
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

SERVER_URL = 'http://192.168.1.24:5000'  # Assurez-vous que c'est correct
MODEL_PATH = 'C:/Users/USER/Desktop/fog/clients/models/model_weights.h5'  # Chemin vers votre modèle

# Charger le modèle d'IA pour la détection d'émotion et notifier le serveur
def load_emotion_model():
    model = load_model(MODEL_PATH)
    print("Modèle chargé avec succès.")
    
    # Envoyer une notification au serveur que le client est prêt à recevoir la partie de l'audio
    notify_server_ready()
    
    return model

# Fonction pour notifier le serveur que le client est prêt
def notify_server_ready():
    try:
        response = requests.post(f'{SERVER_URL}/client-ready', json={'client_id': 'client1'})
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        print("Le serveur a été informé que client1 est prêt à recevoir la partie audio.")
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP lors de la notification du serveur: {http_err}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la notification du serveur: {e}")


# Vérification de la connexion au serveur
def check_server_connection():
    try:
        response = requests.get(SERVER_URL)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        print("Connexion au serveur établie avec succès.")
        return True
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP: {http_err}")
        return False
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Erreur de connexion: {conn_err}")
        return False
    except requests.exceptions.Timeout as timeout_err:
        print(f"Délai d'attente dépassé: {timeout_err}")
        return False
    except requests.exceptions.RequestException as req_err:
        print(f"Erreur de requête: {req_err}")
        return False

# Fonction pour télécharger la partie de l'audio à partir du serveur
def download_audio_part(part_num):
    while True:
        try:
            response = requests.get(f'{SERVER_URL}/send-audio/{part_num}', stream=True)
            if response.status_code == 200:
                audio_file = f'received_audio_part_{part_num}.wav'
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                print(f"Partie {part_num} du fichier audio téléchargée avec succès.")
                return audio_file
            elif response.status_code == 404:
                print(f"Partie {part_num} non encore disponible. Nouvelle tentative dans 5 secondes.")
                time.sleep(5)  # Attendre 5 secondes avant de réessayer
            else:
                print(f"Erreur lors du téléchargement de la partie {part_num} : {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la vérification du fichier: {e}")
            return None

# Prétraitement du fichier audio pour le modèle
def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None, duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

# Simulation de la détection d'émotion à partir du fichier audio
def detect_emotion(model, audio_file, enc):
    processed_audio = preprocess_audio(audio_file)
    predictions = model.predict(processed_audio)
   
    # Afficher les valeurs brutes de la prédiction
    print(f"Valeurs de prédiction (brutes) : {predictions}")
   
    # Utiliser l'encodeur pour inverser la transformation OneHot et obtenir l'étiquette prédite
    predicted_label = enc.inverse_transform(predictions)
    emotion = predicted_label[0][0]
    print(f"Émotion prédite : {emotion}")
    return emotion

# Envoi du résultat au serveur
def send_result(part_num, emotion):
    result = {'part_num': part_num, 'emotion': emotion}
    print(f"Envoi du résultat: {result}")  # Afficher le résultat envoyé
    try:
        response = requests.post(f'{SERVER_URL}/receive-result', json=result)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        print(f"Résultat pour la partie {part_num} envoyé avec succès.")
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP lors de l'envoi du résultat: {http_err}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'envoi du résultat: {e}")

if __name__ == '__main__':
    # Vérifier la connexion au serveur
    if check_server_connection():
        # Charger le modèle d'émotion
        emotion_model = load_emotion_model()
       
        # Créer ou charger l'encodeur OneHotEncoder avec les labels d'émotion
        emotion_labels = ['happy', 'sad', 'angry', 'fear', 'neutral', 'disgust', 'pleasant surprise']
        enc = OneHotEncoder()
        enc.fit(pd.DataFrame(emotion_labels))
       
        # Télécharger et prédire pour la partie 1
        audio_file_part1 = download_audio_part(1)
        if audio_file_part1:
            # Détection d'émotion
            emotion_part1 = detect_emotion(emotion_model, audio_file_part1, enc)
            # Envoi du résultat au serveur
            send_result(1, emotion_part1)
