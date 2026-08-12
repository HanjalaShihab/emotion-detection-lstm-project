"""
Train an LSTM model for 13-class text emotion classification.

Dataset columns:
    tweet_id, sentiment, content

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
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Configuration
CSV_PATH = "emotion.csv"
VOCAB_SIZE = 10000
MAX_LEN = 40
EMBED_DIM = 64
LSTM_UNITS = 64
EPOCHS = 15
BATCH_SIZE = 64

# Load dataset
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)

df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""].reset_index(drop=True)

print(f"Total samples: {len(df)}")
print("\nEmotion distribution:")
print(df["emotion"].value_counts())

# Get classes automatically
CLASS_NAMES = sorted(df["emotion"].unique())
NUM_CLASSES = len(CLASS_NAMES)

print(f"\nTotal classes: {NUM_CLASSES}")
print("Classes:", CLASS_NAMES)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning text...")
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emotion"])

print("\nEncoded classes:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

# Train/test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train_text)}")
print(f"Testing samples: {len(X_test_text)}")

# Tokenization
print("\nTokenizing text...")
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Padding
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print(f"Training input shape: {X_train_pad.shape}")
print(f"Testing input shape: {X_test_pad.shape}")

# Build LSTM model
print("\nBuilding LSTM model...")

vocab_size = min(
    VOCAB_SIZE,
    len(tokenizer.word_index) + 1
)

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM
    ),
    SpatialDropout1D(0.2),
    LSTM(
        LSTM_UNITS,
        dropout=0.2,
        recurrent_dropout=0.1
    ),
    Dropout(0.4),
    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.4),
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, MAX_LEN))
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.00001,
    verbose=1
)

# Train model
print("\nStarting training...")

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test,
    verbose=1
)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save training graph
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(
    history.history["accuracy"],
    label="Training"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    history.history["loss"],
    label="Training"
)
plt.plot(
    history.history["val_loss"],
    label="Validation"
)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.close()

print("\nSaved training graph: training_history.png")

# Save model
model.save("../backend/emotion_model.keras")

# Save tokenizer
with open("../backend/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label classes
with open("../backend/label_classes.pkl", "wb") as f:
    pickle.dump(
        list(label_encoder.classes_),
        f
    )

print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")
print("\nSaved files:")
print("../backend/emotion_model.keras")
print("../backend/tokenizer.pkl")
print("../backend/label_classes.pkl")
print("training_history.png")