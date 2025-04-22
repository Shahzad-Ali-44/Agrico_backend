# 🌾 Agrico AI - FastAPI Server

**Agrico AI** is a FastAPI-based backend service that detects rice crop diseases from leaf images using a pre-trained deep learning model. It provides detailed information about the detected disease, including symptoms and treatment recommendations. The model is loaded directly from the HuggingFace Hub.



## 🚀 Features

- Predicts 9 common rice leaf diseases from images.
- Loads Keras model from [Hugging Face Hub](https://huggingface.co/ShahzadAli44/rice_cnn).
- Returns disease name, confidence score, symptoms, and treatment.
- Built with `FastAPI` and supports CORS for easy frontend integration.
- Accepts image uploads via `/predict` POST endpoint.



## 🧠 Supported Rice Diseases

- Bacterial Leaf Blight
- Brown Spot
- Leaf Blast
- Leaf Scald
- Narrow Brown Spot
- Rice Hispa
- Sheath Blight
- Tungro
- Healthy (No disease)



## 📦 Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Keras
- PIL (Pillow)
- NumPy
- Huggingface Hub

Install dependencies:

```bash
pip install fastapi uvicorn keras pillow numpy huggingface_hub
```



## 🔌 API Endpoints

### `GET /`

Returns a simple message to confirm the API is live.

**Response:**
```json
{ "message": "Agrico API is live!" }
```



### `POST /predict`

Uploads an image and returns prediction results.

**Request:**

- Content-Type: `multipart/form-data`
- Form field: `file` (image)

**Response:**
```json
{
  "class": "brown_spot",
  "confidence": 0.91,
  "symptoms": "Small brown spots appear on leaves, which later turn yellow.",
  "treatment": "Use potassium-rich fertilizers. Spray Mancozeb fungicide. Maintain good drainage to avoid water stress."
}
```

If the prediction confidence is low:
```json
{ "error": "The uploaded image does not appear to be a rice crop leaf." }
```



## 🧠 Model Info

The model is automatically downloaded from HuggingFace:

```python
model_path = hf_hub_download(
    repo_id="ShahzadAli44/rice_cnn",
    filename="rice_cnn_model.keras"
)
```



## 🖼️ How It Works

1. Upload a rice leaf image via the `/predict` endpoint.
2. The image is preprocessed and passed to the CNN model.
3. The model returns predictions across 9 classes.
4. The API responds with the predicted class and related info.



## 🛡️ CORS Enabled

All origins are allowed for easy frontend integration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```



## 🔧 Run the API

Start the FastAPI app using Uvicorn:

```bash
uvicorn main:app --reload
```

Then open your browser or use Postman to test at:
```
http://127.0.0.1:8000
```



## 👨‍💻 Author

Built with ❤️ by [Shahzad Ali](https://shahzadali.vercel.app)



## License

This project is licensed under the [MIT License](LICENSE).



