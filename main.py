import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Pour sauvegarder le modèle
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

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


def extract_features(image_path, include_contrast=True):
    """Extrait les caractéristiques de couleur (dominante et moyenne) et de texture (contraste) d'une image."""
    image = Image.open(image_path).convert("RGB")
    dominant_color = get_dominant_color(image)
    average_color = get_average_color(image)

    features = list(dominant_color) + list(average_color)

    if include_contrast:
        # Convertir l'image en niveaux de gris pour la matrice de co-occurrence
        gray_image = np.array(image.convert("L"))

        # Calculer les matrices de co-occurrence pour les directions 0, 90, 180, 270° et distances 2 et 50 pixels
        distances = [2, 50]
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

        # Calculer le contraste pour chaque combinaison de distance et d'angle
        contrast = graycoprops(glcm, 'contrast')
        contrast_features = contrast.flatten().tolist()

        features += contrast_features

    return features


def create_dataset(include_contrast=True):
    """Crée un dataset d'entraînement avec les images classifiées."""
    data = []

    for label in LABELS:
        train_path = os.path.join(DATASET_DIR, label, "train")
        if not os.path.exists(train_path):
            continue  # Ignore si le dossier n'existe pas

        image_files = [f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_name in tqdm(image_files, desc=f"Processing {label}", unit="image"):
            image_path = os.path.join(train_path, image_name)
            features = extract_features(image_path, include_contrast)
            data.append(features + [label])

    columns = ["dom_R", "dom_G", "dom_B", "avg_R", "avg_G", "avg_B"]
    if include_contrast:
        columns += [f"contrast_{d}_{a}" for d in [2, 50] for a in [0, 90, 180, 270]]
    columns += ["label"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(CSV_FILE, index=False)
    print(f"Dataset enregistré sous {CSV_FILE}")

def create_features_dataset(include_contrast=True):
    """Crée un dataset de features avec les images de test classifiées."""
    data = []

    for label in LABELS:
        test_path = os.path.join(DATASET_DIR, label, "test")
        if not os.path.exists(test_path):
            continue  # Ignore si le dossier n'existe pas

        image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_name in tqdm(image_files, desc=f"Processing {label}", unit="image"):
            image_path = os.path.join(test_path, image_name)
            features = extract_features(image_path, include_contrast)
            data.append(features + [label, "test"])

    columns = ["dom_R", "dom_G", "dom_B", "avg_R", "avg_G", "avg_B"]
    if include_contrast:
        columns += [f"contrast_{d}_{a}" for d in [2, 50] for a in [0, 90, 180, 270]]
    columns += ["label", "subset"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("features_dataset.csv", index=False)
    print("Features dataset enregistré sous 'features_dataset.csv'")

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

def evaluate_model():
    """Évalue le modèle KNN en affichant une matrice de confusion normalisée, les précisions pour chaque classe, une visualisation des features, et les distances intra/inter-classe."""
    if not os.path.exists(MODEL_FILE):
        print("Le modèle n'est pas encore entraîné ! Lancez `train_knn_model()`.")
        return

    knn = joblib.load(MODEL_FILE)
    df = pd.read_csv("features_dataset.csv")
    test_df = df[df["subset"] == "test"]
    features_list = test_df.iloc[:, :-2].values
    labels_list = test_df["label"].values

    y_true = labels_list
    y_pred = knn.predict(features_list)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    precision = precision_score(y_true, y_pred, labels=LABELS, average=None)

    # Afficher la matrice de confusion normalisée
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Afficher les précisions pour chaque classe
    for label, prec in zip(LABELS, precision):
        print(f"Précision pour {label}: {prec:.2f}")

    # Calculer les distances intra-classe et inter-classe
    intra_class_distances = []
    inter_class_distances = []

    for label in LABELS:
        class_features = features_list[labels_list == label]
        other_features = features_list[labels_list != label]

        if len(class_features) > 1:
            intra_distances = pairwise_distances(class_features)
            intra_class_distances.append(intra_distances.mean())

        if len(other_features) > 0:
            inter_distances = pairwise_distances(class_features, other_features)
            inter_class_distances.append(inter_distances.mean())

    avg_intra_class_distance = np.mean(intra_class_distances)
    avg_inter_class_distance = np.mean(inter_class_distances)

    print(f"Distance intra-classe moyenne: {avg_intra_class_distance:.2f}")
    print(f"Distance inter-classe moyenne: {avg_inter_class_distance:.2f}")

    # Visualisation des features en 2D avec PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_list)
    plt.figure(figsize=(10, 7))
    for label in LABELS:
        indices = [i for i, l in enumerate(labels_list) if l == label]
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Visualization of Features using PCA')
    plt.legend()
    plt.show()

    # Visualisation des features en 3D avec t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    features_3d = tsne.fit_transform(features_list)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for label in LABELS:
        indices = [i for i, l in enumerate(labels_list) if l == label]
        ax.scatter(features_3d[indices, 0], features_3d[indices, 1], features_3d[indices, 2], label=label)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D Visualization of Features using t-SNE')
    plt.legend()
    plt.show()


# --- Exécution ---
# Étape 1 : Construire le dataset (exécuter une seule fois après avoir classé les images)
#create_dataset(True)

# Étape 1 bis : Construire le dataset de features
#create_features_dataset(True)

# Étape 2 : Entraîner le modèle (à faire une seule fois ou après mise à jour du dataset)
#train_knn_model()

# Étape 3 : Classifier une nouvelle image
#classify_image("dataset/foret/test/0A5KPRVRPPEH.jpg")

# Étape 4 : Évaluer le modèle
evaluate_model()