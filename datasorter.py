import os
import cv2
import shutil

# Dossiers de destination
dataset_dir = "dataset"
categories = {
    ord('a'): "foret",
    ord('z'): "ocean",
    ord('e'): "plage",
    ord('r'): "desert",
    ord('t'): "montagne",
    ord('y'): "caca"
}

# Création des sous-dossiers s'ils n'existent pas
for folder in categories.values():
    os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)

# Récupération des images du dataset
images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not images:
    print("Aucune image trouvée dans le dossier 'dataset'.")
    exit()

# Parcourir et afficher les images
for image_name in images:
    image_path = os.path.join(dataset_dir, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Impossible d'ouvrir {image_name}")
        continue

    # Affichage de l'image
    cv2.imshow("Classification des images", img)
    print(f"Image en cours : {image_name} - Appuyez sur A/Z/E/R/T/Y pour la classer")

    key = cv2.waitKey(0)  # Attendre une touche

    if key in categories:
        destination = os.path.join(dataset_dir, categories[key], image_name)
        shutil.move(image_path, destination)
        print(f"{image_name} déplacée vers {categories[key]}")

    elif key == 27:  # Touche Échap (ESC) pour quitter
        print("Arrêt du programme.")
        break

cv2.destroyAllWindows()
