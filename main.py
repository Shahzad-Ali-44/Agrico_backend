import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import logging
from huggingface_hub import hf_hub_download
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = hf_hub_download(repo_id="ShahzadAli44/rice_cnn", filename="rice_cnn_model.keras")


try:
    MODEL = keras.saving.load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    MODEL = None

CLASS_NAMES = [
    "bacterial_leaf_blight", "brown_spot", "healthy", "leaf_blast",
    "leaf_scald", "narrow_brown_spot", "rice_hispa", "sheath_blight", "tungro"
]

DISEASE_DETAILS = {
    "bacterial_leaf_blight": {
        "symptoms": "Leaves turn yellow with wet-looking spots that spread and dry out.",
        "treatment": "Use disease-resistant rice seeds. Spray copper-based fungicides. Apply balanced fertilizers with nitrogen and phosphorus."
    },
    "brown_spot": {
        "symptoms": "Small brown spots appear on leaves, which later turn yellow.",
        "treatment": "Use potassium-rich fertilizers. Spray Mancozeb fungicide. Maintain good drainage to avoid water stress."
    },
    "healthy": {
        "symptoms": "Leaves are green and strong, with no signs of disease.",
        "treatment": "No treatment needed. Keep the soil healthy by using compost and balanced fertilizers."
    },
    "leaf_blast": {
        "symptoms": "Leaves get white or gray spots that spread and kill the leaf.",
        "treatment": "Spray Tricyclazole fungicide. Keep the right water level in the field. Use silica-based fertilizers to strengthen plants."
    },
    "leaf_scald": {
        "symptoms": "Leaf edges turn yellow or brown, and the leaf dries up.",
        "treatment": "Use resistant rice varieties. Avoid too much nitrogen fertilizer. Spray Propiconazole fungicide if needed."
    },
    "narrow_brown_spot": {
        "symptoms": "Thin, dark brown streaks appear on leaves.",
        "treatment": "Reduce plant overcrowding. Spray Propiconazole fungicide. Use potassium and phosphorus fertilizers to improve plant health."
    },
    "rice_hispa": {
        "symptoms": "Leaves get white scars and small holes due to insect feeding.",
        "treatment": "Remove infected leaves. Spray Chlorpyrifos insecticide. Keep fields clean to reduce insect attacks."
    },
    "sheath_blight": {
        "symptoms": "White or gray patches appear on the lower part of the plant, leading to weak stems.",
        "treatment": "Keep enough space between plants. Apply Azoxystrobin fungicide. Use compost and phosphorus-rich fertilizers."
    },
    "tungro": {
        "symptoms": "Plants grow slowly, and leaves turn yellow or orange.",
        "treatment": "Use virus-free seedlings. Spray insecticides like Imidacloprid to control pests. Apply nitrogen fertilizers to strengthen plants."
    }
}


def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  
        logging.debug(f"Image size: {image.size}")
        return np.array(image)
    except Exception as e:
        logging.error("Error reading image file: %s", str(e))
        raise ValueError("Invalid image data")


@app.get("/")
def home():
    return {"message": "Agrico API is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model is not loaded properly."}

    try:
        image_data = await file.read()
        image = read_file_as_image(image_data)
        img_batch = np.expand_dims(image, 0)
        logging.debug(f"Image batch shape: {img_batch.shape}")
        
        predictions = MODEL.predict(img_batch)
        logging.debug(f"Predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        if confidence < 0.5:
            return {"error": "The uploaded image does not appear to be a rice crop leaf."}
        disease_details = DISEASE_DETAILS.get(predicted_class, {})
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'symptoms': disease_details.get("symptoms", "No symptoms available."),
            'treatment': disease_details.get("treatment", "No treatment information available.")
        }
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return {"error": "An error occurred while processing the image."}
