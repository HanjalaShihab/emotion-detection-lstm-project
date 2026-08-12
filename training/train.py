"""
Train an LSTM model for 6-class text emotion classification.

Classes: joy, sadness, anger, fear, love, surprise

Expects a CSV file at training/emotion.csv with two columns:
    text,emotion

See training/README.md for where to get this dataset.

Run:
    python train.py
"""

import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Config
# -----------------------------
CSV_PATH = "emotion.csv"
VOCAB_SIZE = 10000
MAX_LEN = 40
EMBED_DIM = 64
LSTM_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 64

CLASS_NAMES = ["joy", "sadness", "anger", "fear", "love", "surprise"]

# -----------------------------
# 2. Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "emotion"])

# Keep only the six emotions we care about
df["emotion"] = df["emotion"].str.lower().str.strip()
df = df[df["emotion"].isin(CLASS_NAMES)].reset_index(drop=True)

print(f"Total samples: {len(df)}")
print(df["emotion"].value_counts())


# -----------------------------
# 3. Clean text
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)             # keep letters only
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text


df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# 4. Encode labels
# -----------------------------
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)  # fixes label order to CLASS_NAMES
y = label_encoder.transform(df["emotion"])

# -----------------------------
# 5. Train / test split
# -----------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Tokenize
# -----------------------------
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# -----------------------------
# 7. Pad sequences
# -----------------------------
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# -----------------------------
# 8. Build LSTM model
# -----------------------------
vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN, mask_zero=True),
    LSTM(LSTM_UNITS),
    Dense(32, activation="relu"),
    Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 9. Train
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
)

# -----------------------------
# 10. Evaluate
# -----------------------------
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot accuracy / loss curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Saved training curves to training_history.png")

# -----------------------------
# 11. Save model, tokenizer, label classes
# -----------------------------
model.save("../backend/emotion_model.keras")

with open("../backend/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("../backend/label_classes.pkl", "wb") as f:
    pickle.dump(list(label_encoder.classes_), f)

print("\nSaved:")
print("  ../backend/emotion_model.keras")
print("  ../backend/tokenizer.pkl")
print("  ../backend/label_classes.pkl")
