import os
import shutil
import random

# Dossier contenant les images
DATASET_DIR = "dataset"
SUBDIRS = ["desert", "foret", "montagne", "ocean", "plage"]

# Pourcentage de données d'entraînement
TRAIN_RATIO = 0.8

def split_dataset():
    for subdir in SUBDIRS:
        subdir_path = os.path.join(DATASET_DIR, subdir)
        if not os.path.exists(subdir_path):
            continue

        # Créer les sous-dossiers "train" et "test"
        train_path = os.path.join(subdir_path, "train")
        test_path = os.path.join(subdir_path, "test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Récupérer toutes les images du sous-dossier
        images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Mélanger les images
        random.shuffle(images)

        # Calculer le nombre d'images pour l'entraînement
        train_size = int(len(images) * TRAIN_RATIO)

        # Déplacer les images dans les sous-dossiers "train" et "test"
        for i, image_name in enumerate(images):
            src_path = os.path.join(subdir_path, image_name)
            if i < train_size:
                dest_path = os.path.join(train_path, image_name)
            else:
                dest_path = os.path.join(test_path, image_name)
            shutil.move(src_path, dest_path)

        print(f"Images from {subdir} split into train and test sets.")

# Exécuter la fonction
split_dataset()