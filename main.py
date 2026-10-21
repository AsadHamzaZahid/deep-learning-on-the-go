from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi import FastAPI, HTTPException
import numpy as np
from keras.models import load_model
import pickle
import re

model_path = "Artifacts/BiGru_model.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Global runtime storage
model_dict = {}


@asynccontextmanager
async def life_span(app: FastAPI):
    # Load assets securely on startup
    model_dict["BiGru"] = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        model_dict['Tokenizer'] = pickle.load(file)

    print("Model loaded successfully")
    yield
    model_dict.clear()

app = FastAPI(lifespan=life_span)

# --- PYDANTIC SCHEMAS ---


class UserInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description='The sentence to analyze')

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this park so much that I have tears in my eyes"
            }
        }


class HealthyConnections(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    probabilities: dict[str, float]

# --- TEXT HELPER FUNCTION ---


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub("'", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Changed from clean out all spaces to keeping standard spacing between words
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- API ROUTES ---


@app.get("/", response_model=HealthyConnections)
def hello():
    # Model is loaded if keys are present in our global dictionary
    is_loaded = "BiGru" in model_dict and "Tokenizer" in model_dict
    return HealthyConnections(status="Connection is loaded", model_loaded=is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: UserInput):  # Accept Pydantic model payload here

    bigru = model_dict.get("BiGru")
    tokens = model_dict.get("Tokenizer")

    if bigru is None or tokens is None:
        raise HTTPException(
            status_code=503, detail="Model assets are not initialized or loaded yet")

    # 1. Clean the input string extracted from Pydantic
    cleaned_text = clean_text(payload.text)

    # 2. Tokenize text using the pickle loaded tokenizer
    sequences = tokens.texts_to_sequences([cleaned_text])

    # 3. Correct usage of pad_sequences function
    padded_sequences = pad_sequences(
        sequences, maxlen=50, padding="post", truncating='post'
    )

    # 4. Infer predictions
    probabilities = bigru.predict(padded_sequences)[0]
    top_emotion_index = int(np.argmax(probabilities))

    # 5. Zip probabilities properly with global emotions string array labels
    all_probabilities = {
        label: float(prob) for label, prob in zip(emotions, probabilities)
    }

    # 6. Return response schema perfectly matching prediction_response fields
    return PredictionResponse(
        text=payload.text,
        emotion=emotions[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        probabilities=all_probabilities
    )
