import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Pour sauvegarder le modèle

# Dossier contenant les images d'entraînement
DATASET_DIR = "dataset"
LABELS = ["foret", "ocean", "plage", "desert", "montagne"]
CSV_FILE = "colors_dataset.csv"
MODEL_FILE = "knn_model.pkl"

def get_dominant_color(image):
    """Retourne la couleur dominante de l'image."""
    pixels = np.array(image).reshape(-1, 3)
    most_common = Counter(map(tuple, pixels)).most_common(1)
    return most_common[0][0] if most_common else (0, 0, 0)


def get_average_color(image):
    """Retourne la couleur moyenne de l'image."""
    pixels = np.array(image)
    avg_color = pixels.mean(axis=(0, 1))
    return tuple(map(int, avg_color))


def extract_features(image_path):
    """Extrait les caractéristiques de couleur (dominante et moyenne) d'une image."""
    image = Image.open(image_path).convert("RGB")
    dominant_color = get_dominant_color(image)
    average_color = get_average_color(image)
    return list(dominant_color) + list(average_color)


def create_dataset():
    """Crée un dataset d'entraînement avec les images classifiées."""
    data = []

    for label in LABELS:
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.exists(label_path):
            continue  # Ignore si le dossier n'existe pas

        for image_name in os.listdir(label_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(label_path, image_name)
                features = extract_features(image_path)
                data.append(features + [label])

    df = pd.DataFrame(data, columns=["dom_R", "dom_G", "dom_B", "avg_R", "avg_G", "avg_B", "label"])
    df.to_csv(CSV_FILE, index=False)
    print(f"Dataset enregistré sous {CSV_FILE}")


def train_knn_model():
    """Entraîne un modèle KNN avec le dataset de couleurs."""
    df = pd.read_csv(CSV_FILE)
    X = df.iloc[:, :-1].values  # Features (couleurs)
    y = df["label"].values  # Labels (catégories)

    knn = KNeighborsClassifier(n_neighbors=3)  # K=3
    knn.fit(X, y)

    # Sauvegarder le modèle entraîné
    joblib.dump(knn, MODEL_FILE)
    print(f"Modèle KNN entraîné et sauvegardé sous {MODEL_FILE}")


def classify_image(image_path):
    """Classifie une image en utilisant le modèle KNN."""
    if not os.path.exists(MODEL_FILE):
        print("Le modèle n'est pas encore entraîné ! Lancez `train_knn_model()`.")
        return

    knn = joblib.load(MODEL_FILE)
    features = np.array(extract_features(image_path)).reshape(1, -1)
    prediction = knn.predict(features)[0]

    # Afficher l'image avec sa classification
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Classification : {prediction}")
    plt.show()
    print(f"L'image est classée comme : {prediction}")

# --- Exécution ---
# Étape 1 : Construire le dataset (exécuter une seule fois après avoir classé les images)
#create_dataset()

# Étape 2 : Entraîner le modèle (à faire une seule fois ou après mise à jour du dataset)
#train_knn_model()

# Étape 3 : Classifier une nouvelle image
#classify_image("dataset/00000026_(4).jpg")
