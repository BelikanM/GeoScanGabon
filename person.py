import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from sklearn.ensemble import RandomForestClassifier
import base64
import time
import logging
import traceback

app = Flask(__name__)
CORS(app)

# Configuration du logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dossier pour stocker les modèles
MODELS_DIR = "joblib"
os.makedirs(MODELS_DIR, exist_ok=True)

# Chemins des modèles
PEOPLE_COUNTER_MODEL = os.path.join(MODELS_DIR, "people_counter.joblib")
EMOTION_MODEL = os.path.join(MODELS_DIR, "emotion_classifier.joblib")
POSTURE_MODEL = os.path.join(MODELS_DIR, "posture_classifier.joblib")
OBJECT_MODEL = os.path.join(MODELS_DIR, "object_detector.joblib")

# Configuration des paramètres du HOGdescriptor
winSize = (128, 128)  # Taille des images redimensionnées
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

EXPECTED_FEATURE_LENGTH = 3780  # Taille fixe attendue par modèles

def extract_features(image):
    """
    Extract HOG features from the image and normalize length to EXPECTED_FEATURE_LENGTH
    """
    try:
        image_resized = cv2.resize(image, winSize)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        features = hog.compute(gray)
        features_flat = features.flatten()

        logger.debug(f"Extracted features length before normalization: {len(features_flat)}")

        # Adapter la taille des features
        if len(features_flat) > EXPECTED_FEATURE_LENGTH:
            features_flat = features_flat[:EXPECTED_FEATURE_LENGTH]
        elif len(features_flat) < EXPECTED_FEATURE_LENGTH:
            features_flat = np.pad(features_flat, (0, EXPECTED_FEATURE_LENGTH - len(features_flat)), 'constant')

        logger.debug(f"Extracted features length after normalization: {len(features_flat)}")
        return features_flat
    except Exception as e:
        logger.error(f"Erreur dans extract_features: {e}")
        logger.error(traceback.format_exc())
        raise

def create_and_save_models():
    try:
        if not os.path.exists(PEOPLE_COUNTER_MODEL):
            logger.info("Création du modèle people_counter")
            model = RandomForestClassifier(n_estimators=100)
            X_dummy = np.random.rand(100, EXPECTED_FEATURE_LENGTH)
            y_dummy = np.random.randint(0, 5, 100)
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, PEOPLE_COUNTER_MODEL)
            logger.info(f"Modèle de comptage créé : {PEOPLE_COUNTER_MODEL}")

        if not os.path.exists(EMOTION_MODEL):
            logger.info("Création du modèle emotion_classifier")
            model = RandomForestClassifier(n_estimators=100)
            X_dummy = np.random.rand(100, EXPECTED_FEATURE_LENGTH)
            y_dummy = np.random.choice(['joie', 'tristesse', 'colère', 'surprise', 'neutre'], 100)
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, EMOTION_MODEL)
            logger.info(f"Modèle émotion créé : {EMOTION_MODEL}")

        if not os.path.exists(POSTURE_MODEL):
            logger.info("Création du modèle posture_classifier")
            model = RandomForestClassifier(n_estimators=100)
            X_dummy = np.random.rand(100, EXPECTED_FEATURE_LENGTH)
            y_dummy = np.random.choice(['assis', 'debout', 'courbé', 'allongé', 'acrobatique'], 100)
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, POSTURE_MODEL)
            logger.info(f"Modèle posture créé : {POSTURE_MODEL}")

        if not os.path.exists(OBJECT_MODEL):
            logger.info("Création du modèle object_detector")
            model = RandomForestClassifier(n_estimators=100)
            X_dummy = np.random.rand(100, EXPECTED_FEATURE_LENGTH)
            y_dummy = np.random.choice(['personne', 'arbre', 'voiture', 'bâtiment', 'animal'], 100)
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, OBJECT_MODEL)
            logger.info(f"Modèle objet créé : {OBJECT_MODEL}")
    except Exception as e:
        logger.error(f"Erreur dans create_and_save_models: {e}")
        logger.error(traceback.format_exc())
        raise

def load_models():
    try:
        logger.debug("Chargement des modèles")
        people_counter = joblib.load(PEOPLE_COUNTER_MODEL)
        emotion_classifier = joblib.load(EMOTION_MODEL)
        posture_classifier = joblib.load(POSTURE_MODEL)
        object_detector = joblib.load(OBJECT_MODEL)
        logger.debug("Modèles chargés avec succès")
        return {
            'people_counter': people_counter,
            'emotion_classifier': emotion_classifier,
            'posture_classifier': posture_classifier,
            'object_detector': object_detector
        }
    except Exception as e:
        logger.error(f"Erreur dans load_models: {e}")
        logger.error(traceback.format_exc())
        raise

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not request.is_json:
        logger.warning("Requête sans JSON")
        return jsonify({'error': 'Requête invalide : JSON attendu'}), 400

    if 'image' not in request.json:
        logger.warning("Aucune image fournie")
        return jsonify({'error': 'Aucune image fournie'}), 400

    try:
        raw_image_data = request.json['image']
        logger.debug(f"Taille de l'image reçue : {len(raw_image_data)} caractères")

        if ',' in raw_image_data:
            image_data = raw_image_data.split(',')[1]
        else:
            image_data = raw_image_data

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Erreur : l'image n'a pas pu être décodée")
            return jsonify({'error': 'Image invalide'}), 400

        features = extract_features(image)
        models = load_models()

        people_count = models['people_counter'].predict([features])[0]

        emotions = []
        postures = []
        objects = []

        for i in range(min(int(people_count), 3)):
            emotion = models['emotion_classifier'].predict([features])[0]
            posture = models['posture_classifier'].predict([features])[0]
            emotions.append({'id': i, 'emotion': emotion, 'confidence': float(np.random.uniform(0.7, 0.95))})
            postures.append({'id': i, 'posture': posture, 'confidence': float(np.random.uniform(0.7, 0.95))})

        for i in range(int(np.random.randint(2, 5))):
            object_class = models['object_detector'].predict([features])[0]
            quality = np.random.choice(['neuf', 'usé', 'abîmé', 'en bon état'])
            objects.append({
                'id': i,
                'class': object_class,
                'confidence': float(np.random.uniform(0.6, 0.9)),
                'position': {
                    'x': int(np.random.uniform(0, image.shape[1])),
                    'y': int(np.random.uniform(0, image.shape[0])),
                    'width': int(np.random.uniform(50, 150)),
                    'height': int(np.random.uniform(50, 150))
                },
                'attributes': {
                    'qualité': quality,
                    'taille': np.random.choice(['petit', 'moyen', 'grand']),
                    'couleur': np.random.choice(['rouge', 'bleu', 'vert', 'jaune', 'noir', 'blanc'])
                }
            })

        logger.info(f"Analyse terminée : {people_count} personnes détectées")
        return jsonify({
            'timestamp': time.time(),
            'analysis': {
                'people_count': int(people_count),
                'people': {
                    'emotions': emotions,
                    'postures': postures
                },
                'environment': {
                    'objects': objects
                }
            }
        })
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse : {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'models_loaded': [
            'people_counter',
            'emotion_classifier',
            'posture_classifier',
            'object_detector'
        ]
    })

if __name__ == '__main__':
    try:
        logger.info("Initialisation des modèles")
        create_and_save_models()
    except Exception as e:
        logger.critical(f"Erreur critique lors de la création des modèles : {e}")
        logger.critical(traceback.format_exc())
        exit(1)

    logger.info("Démarrage du serveur Flask")
    app.run(host='0.0.0.0', port=9000, debug=True)

