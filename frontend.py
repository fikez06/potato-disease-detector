import streamlit as st
from PIL import Image
import requests
import io

# ------------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Potato Diseases Detector",
    page_icon="potato",
    layout="centered"
)

st.title("Potato Leaf Disease Detection")
st.write("Charge une photo d'une feuille de pomme de terre et je te dis si elle est malade")

# ------------------------------------------------------------------
# Upload de l'image
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choisis une image de feuille de pomme de terre...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Affichage de l'image uploadée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_column_width=True)
    st.write("")

    # Préparation des colonnes pour résultat côte à côte
    col1, col2 = st.columns([1, 1])

    with st.spinner("Analyse en cours..."):
        # Envoi vers ton backend FastAPI
        bytes_data = uploaded_file.getvalue()
        files = {"file": bytes_data}
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                predicted_class = result["class"]
                confidence = result["confidence"] * 100  # on passe en %

                # Couleurs selon la classe
                if predicted_class == "Healthy":
                    color = "green"
                    emoji = "healthy"
                    message = f"La feuille est **saine** !"
                elif predicted_class == "Early Blight":
                    color = "orange"
                    emoji = "warning"
                    message = f"Attention : **Early Blight** détectée"
                else:  # Late Blight
                    color = "red"
                    emoji = "danger"
                    message = f"Attention : **Late Blight** détectée (très contagieuse !)"

                with col1:
                    st.markdown(f"### **{predicted_class}**")
                    st.markdown(message)

                with col2:
                    st.metric(
                        label="Confiance du modèle",
                        value=f"{confidence:.2f}%"
                    )

                    # Barre de progression stylée
                    st.progress(confidence / 100)

                    # Petits conseils selon la maladie
                    if predicted_class == "Early Blight":
                        st.info("Conseil : Applique un fongicide à base de cuivre dès que possible")
                    elif predicted_class == "Late Blight":
                        st.error("URGENT : Retire et brûle les feuilles infectées pour éviter la propagation !")
                    else:
                        st.success("Tout va bien, continue comme ça !")

            else:
                st.error(f"Erreur {response.status_code} : {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter le serveur FastAPI. Vérifie que ton backend est lancé sur http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

# ------------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------------
st.markdown("---")
st.caption("Modèle entraîné sur +4000 images de feuilles de pommes de terre • FastAPI + TensorFlow + Streamlit • 2025")