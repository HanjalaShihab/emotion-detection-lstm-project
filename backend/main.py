"""
FastAPI backend for the Text Emotion Detection project.

Loads the LSTM model trained by training/train.py and serves predictions.
Does NOT train anything — only loads the already-saved model.

Run:
    uvicorn main:app --reload
"""

import re
import pickle

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Config (must match training/train.py)
# -----------------------------
MAX_LEN = 40
EMOJI_MAP = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

# -----------------------------
# Load saved model + tokenizer + labels
# -----------------------------
model = load_model("emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_classes.pkl", "rb") as f:
    class_names = pickle.load(f)  # order matches model output index

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Text Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: TextInput):
    cleaned = clean_text(input_data.text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    probabilities = model.predict(padded, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    all_probabilities = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return {
        "emotion": predicted_emotion,
        "emoji": EMOJI_MAP.get(predicted_emotion, ""),
        "confidence": confidence,
        "probabilities": all_probabilities,
    }
