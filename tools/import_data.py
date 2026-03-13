import s3fs
import os
import zipfile
import json

MON_BUCKET = "omarboumhaousse" 
DOSSIER_CIBLE = "diffusion"

# Créer un dossier 'data' pour y mettre les données
DATA_LOCAL_DIR = "data"
os.makedirs(DATA_LOCAL_DIR, exist_ok=True)

# CONNEXION AU CLOUD

print("Connexion à MinIO...")
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

# TÉLÉCHARGEMENT DU JSON

print("\n Téléchargement de OmniDocBench.json...")
chemin_json_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/OmniDocBench.json"
chemin_json_local = os.path.join(DATA_LOCAL_DIR, "OmniDocBench.json")

if fs.exists(chemin_json_s3):
    fs.get(chemin_json_s3, chemin_json_local)
    
else:
    print("JSON introuvable sur le cloud.")

# FONCTION POUR NETTOYER LES DOSSIERS DOUBLES

def nettoyer_dossier(dossier_cible):
    """
    Si le dossier contient un sous-dossier du même nom (ex: pdfs/pdfs),
    on remonte les fichiers d'un cran et on supprime le sous-dossier vide.
    """
    sous_dossier = os.path.join(dossier_cible, os.path.basename(dossier_cible))
    
    if os.path.exists(sous_dossier):
        fichiers = os.listdir(sous_dossier)
        for fichier in fichiers:
            src = os.path.join(sous_dossier, fichier)
            dst = os.path.join(dossier_cible, fichier)
            os.rename(src, dst)
        # Suppression du dossier maintenant vide
        os.rmdir(sous_dossier)

# TÉLÉCHARGEMENT ET EXTRACTION DES PDFs

print("\n Téléchargement de pdfs.zip...")
chemin_zip_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/pdfs.zip"
chemin_zip_local = os.path.join(DATA_LOCAL_DIR, "pdfs.zip")
dossier_pdf_local = os.path.join(DATA_LOCAL_DIR, "pdfs")

if fs.exists(chemin_zip_s3):
    # Téléchargement
    fs.get(chemin_zip_s3, chemin_zip_local)
    
    # Extraction
    if not os.path.exists(dossier_pdf_local):
        os.makedirs(dossier_pdf_local)
        
    with zipfile.ZipFile(chemin_zip_local, 'r') as zip_ref:
        zip_ref.extractall(dossier_pdf_local)
    
    # NETTOYAGE 
    nettoyer_dossier(dossier_pdf_local)
    
    # Suppression du zip
    os.remove(chemin_zip_local)
    
    
else:
    print("Le fichier pdfs.zip est introuvable sur le cloud.")

# TÉLÉCHARGEMENT ET EXTRACTION DES IMAGES

print("\n Téléchargement de images.zip...")
chemin_zip_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/images.zip"
chemin_zip_local = os.path.join(DATA_LOCAL_DIR, "images.zip")
dossier_images_local = os.path.join(DATA_LOCAL_DIR, "images") # Correction de variable

if fs.exists(chemin_zip_s3):
    # Téléchargement
    fs.get(chemin_zip_s3, chemin_zip_local)
    
    # Extraction
    if not os.path.exists(dossier_images_local):
        os.makedirs(dossier_images_local)
        
    with zipfile.ZipFile(chemin_zip_local, 'r') as zip_ref:
        zip_ref.extractall(dossier_images_local)
    
    # NETTOYAGE
    nettoyer_dossier(dossier_images_local)
    
    # Suppression du zip
    os.remove(chemin_zip_local)
