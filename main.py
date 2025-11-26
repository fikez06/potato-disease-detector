from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras

app = FastAPI ()


MODEL = keras.models.load_model("/Users/ezechielhounnouvi/Documents/Projet DL/Potatos Diseases/models/model_v1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image (data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
  file: UploadFile = File(..., description="Potatoes leaves images")     
):
    image = read_file_as_image (await file.read())
    # Redimensionne si ton modèle attend 224x224 ou 256x256 (adapte selon ton entraînement)
    img = Image.fromarray(image)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0             # Normalisation très courante
    img_batch = np.expand_dims(img_array, 0)
    
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))    # on convertit en float python natif
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    } 

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)