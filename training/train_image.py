"""
Train a small CNN for facial emotion detection from images.

Classes (from folder names): angry, disgusted, fearful, happy, neutral, sad, surprised

Expects:
    image_emotion/train/<class>/*.jpg
    image_emotion/test/<class>/*.jpg

Run:
    python train_image.py
"""

import pickle

import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 1. Config
# -----------------------------
DATA_DIR = "image_emotion"
IMG_SIZE = (48, 48)   # standard FER-style size, grayscale
BATCH_SIZE = 64
EPOCHS = 15

# -----------------------------
# 2. Load images from folders
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize pixel values to [0, 1] and speed up the input pipeline
normalize = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalize(x), y)).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalize(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3. Build a small CNN
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 4. Train
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop],
)

# -----------------------------
# 5. Evaluate
# -----------------------------
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# -----------------------------
# 6. Save model + class names
# -----------------------------
model.save("../backend/emotion_image_model.keras")

with open("../backend/image_labels.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("\nSaved:")
print("  ../backend/emotion_image_model.keras")
print("  ../backend/image_labels.pkl")
