from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import os
import wave
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

last_uploaded_file_path = None  # Chemin du dernier fichier téléchargé
client_results = {}  # Dictionnaire pour stocker les résultats des clients

# Page d'accueil avec formulaire de téléversement
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer l'upload du fichier
@app.route('/upload', methods=['POST'])
def upload_file():
    global last_uploaded_file_path
    if 'file' not in request.files:
        return "Aucun fichier sélectionné", 400

    file = request.files['file']
    if file.filename == '':
        return "Nom de fichier vide", 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        last_uploaded_file_path = filepath
        return render_template('index.html', message="Fichier téléchargé avec succès")

# Route pour diviser l'audio en parties et envoyer la partie demandée
@app.route('/send-audio/<int:part_num>', methods=['GET'])
def send_audio(part_num):
    if not last_uploaded_file_path:
        return jsonify({"error": "Aucun fichier trouvé"}), 404

    try:
        audio = wave.open(last_uploaded_file_path, 'rb')
        frames = audio.getnframes()
        frame_rate = audio.getframerate()
        half_frames = frames // 2

        # Diviser l'audio en deux parties selon la demande
        if part_num == 1:
            audio.setpos(0)
            part_data = audio.readframes(half_frames)
        elif part_num == 2:
            audio.setpos(half_frames)
            part_data = audio.readframes(half_frames)
        else:
            return jsonify({"error": "Numéro de partie invalide"}), 400

        # Sauvegarder la partie audio temporairement
        part_file_path = f"{UPLOAD_FOLDER}/part_{part_num}.wav"
        with wave.open(part_file_path, 'wb') as part_audio:
            part_audio.setnchannels(audio.getnchannels())
            part_audio.setsampwidth(audio.getsampwidth())
            part_audio.setframerate(frame_rate)
            part_audio.writeframes(part_data)

        audio.close()
        return send_file(part_file_path, as_attachment=True)

    except Exception as e:
        print(f"Erreur lors de la division de l'audio : {e}")
        return jsonify({"error": "Erreur lors du traitement de l'audio"}), 500

# Route pour recevoir le résultat d'émotion du client
@app.route('/receive-result', methods=['POST'])
def receive_result():
    global client_results
    try:
        data = request.get_json()
        print(f"Données reçues: {data}")  # Debugging

        part_num = data.get('part_num')
        emotion = data.get('emotion')

        # Vérification que les deux champs existent
        if part_num is not None and emotion is not None:
            client_results[part_num] = emotion  # Stocker le résultat reçu
            print(f"Résultat reçu du client pour la partie {part_num}: {emotion}")

            # Vérifier si les deux résultats ont été reçus
            if len(client_results) == 2:
                # Calcul du résultat final basé sur la majorité
                final_emotion = max(set(client_results.values()), key=list(client_results.values()).count)
                print(f"Résultat final basé sur la majorité: {final_emotion}")
                return jsonify({"message": "Résultat reçu avec succès", "final_emotion": final_emotion}), 200

            return jsonify({"message": "Résultat reçu avec succès, en attente d'autres résultats"}), 200

        return jsonify({"error": "Partie ou émotion manquante"}), 400

    except Exception as e:
        print(f"Erreur dans le traitement des résultats: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500

# Route pour afficher le résultat sur la page result.html
@app.route('/result')
def result_page():
    final_emotion = max(set(client_results.values()), key=list(client_results.values()).count) if client_results else "Résultat en attente"
    return render_template('result.html', emotion=final_emotion)
# Route pour vérifier si le client est prêt
@app.route('/client-ready', methods=['GET', 'POST'])
def client_ready():
    if request.method == 'POST':
        # Récupérer les données envoyées par le client (si applicable)
        data = request.get_json()
        # Envoyer une réponse avec le statut du client
        return jsonify({
            "message": "Client prêt",
            "status": data.get("status", "unknown")
        }), 200
    else:
        # Répondre avec un message simple pour les requêtes GET
        return jsonify({"message": "Client prêt"}), 200
if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)