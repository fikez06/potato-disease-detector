# Potato Leaf Disease Detector

Détection en temps réel des maladies des feuilles de pomme de terre (Early Blight, Late Blight, Healthy) à partir d'une simple photo.

Modèle entraîné avec **TensorFlow/Keras** sur plus de 4000 images → précision > 95 %.

Démo live : https://potato-disease-detector.streamlit.app

## Maladies détectées
- **Early Blight** (Alternariose)
- **Late Blight** (Mildiou) – très contagieux !
- **Healthy** (Feuille saine)

## Fonctionnalités
- Upload d'image via interface web
- Prédiction instantanée avec pourcentage de confiance
- Conseils agricoles adaptés
- Backend FastAPI + Frontend Streamlit
- Déployé gratuitement sur Streamlit Community Cloud

## Comment lancer en local

1. Clone le repo
```bash
git clone https://github.com/fikez06/potato-disease-detector.git
cd potato-disease-detector