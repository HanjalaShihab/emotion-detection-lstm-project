"""
Train a CNN model for facial emotion detection from images.

Classes are automatically loaded from folder names.

Expected structure:
    image_emotion/
        train/
            angry/
            disgusted/
            fearful/
            happy/
            neutral/
            sad/
            surprised/
        test/
            angry/
            disgusted/
            fearful/
            happy/
            neutral/
            sad/
            surprised/

Run:
    python train_image.py
"""

import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# Configuration
DATA_DIR = "image_emotion"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

# Load training images
print("Loading training images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Load test images
print("\nLoading test images...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nTotal classes: {NUM_CLASSES}")

# Count training samples
class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for _, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] += 1

print("\nTraining image distribution:")
for i, name in enumerate(class_names):
    print(f"{name}: {class_counts[i]}")

# Calculate class weights
labels_for_weights = []

for _, labels in train_ds:
    labels_for_weights.extend(labels.numpy())

labels_for_weights = np.array(labels_for_weights)

classes = np.unique(labels_for_weights)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels_for_weights
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.2f}")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

# Normalize images
normalization = layers.Rescaling(1.0 / 255)

# Improve input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Build CNN
print("\nBuilding CNN model...")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    normalization,
    data_augmentation,

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.40),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, IMG_SIZE[0], IMG_SIZE[1], 1))

model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# Train
print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating CNN...")

loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n==============================")
print("CNN MODEL RESULTS")
print("==============================")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
model.save("../backend/emotion_image_model.keras")

# Save labels
with open("../backend/image_labels.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("\n==============================")
print("CNN TRAINING COMPLETED")
print("==============================")

print("\nSaved:")
print("../backend/emotion_image_model.keras")
print("../backend/image_labels.pkl")