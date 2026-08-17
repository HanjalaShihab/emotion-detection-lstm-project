import os
import pickle
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Dropout, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG
# ----------------------------
DATASET_DIR = "emotion detection dataset main"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = 48
BATCH_SIZE = 256         # larger batch = fewer Python/BLAS calls = faster on CPU
EPOCHS = 40              # early stopping will cut this short if needed
SEED = 42
L2_REG = 1e-4

AUTOTUNE = tf.data.AUTOTUNE

# CPU thread tuning - use all available cores for data pipeline + ops
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

print("Loading image dataset with tf.data (fast CPU pipeline)...")

# ----------------------------
# CLASS NAMES (sorted, matches Keras convention)
# ----------------------------
class_names = sorted(os.listdir(TRAIN_DIR))
num_classes = len(class_names)
class_to_idx = {name: i for i, name in enumerate(class_names)}
print("Classes:", class_names)
print("Number of classes:", num_classes)


def list_files_and_labels(directory):
    filepaths, labels = [], []
    for cname in class_names:
        cdir = os.path.join(directory, cname)
        for fp in glob.glob(os.path.join(cdir, "*")):
            filepaths.append(fp)
            labels.append(class_to_idx[cname])
    return filepaths, labels


train_files, train_labels = list_files_and_labels(TRAIN_DIR)
val_files, val_labels = list_files_and_labels(VALIDATION_DIR)
test_files, test_labels = list_files_and_labels(TEST_DIR)

print("Training images:", len(train_files))
print("Validation images:", len(val_files))
print("Testing images:", len(test_files))


def decode_img(filepath, label, augment=False):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        # small random rotation via padding + random crop (cheap on CPU)
        img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 6, IMG_SIZE + 6)
        img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 1])

    label = tf.one_hot(label, num_classes)
    return img, label


def make_dataset(filepaths, labels, augment, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda f, l: decode_img(f, l, augment=augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_files, train_labels, augment=True, shuffle=True)
val_ds = make_dataset(val_files, val_labels, augment=False, shuffle=False)
test_ds = make_dataset(test_files, test_labels, augment=False, shuffle=False)

# ----------------------------
# CLASS WEIGHTS (handles imbalance, e.g. 'disgust' having fewer images)
# ----------------------------
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weight_dict)

# ----------------------------
# MODEL
# ----------------------------
print("Building CNN model...")


def conv_block(x_filters, kernel_size, dropout_rate):
    # single conv per block (not double) - keeps CPU training time reasonable
    return [
        Conv2D(x_filters, kernel_size, padding="same", kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(2, 2),
        Dropout(dropout_rate),
    ]


model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    *conv_block(64, (3, 3), 0.20),
    *conv_block(128, (3, 3), 0.25),
    *conv_block(256, (3, 3), 0.30),
    *conv_block(256, (3, 3), 0.35),

    GlobalAveragePooling2D(),

    Dense(256, kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
os.makedirs("../backend", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "../backend/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ----------------------------
# TRAIN
# ----------------------------
print("Starting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# ----------------------------
# EVALUATE
# ----------------------------
print("Evaluating model...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------------
# CLASSIFICATION REPORT + CONFUSION MATRIX
# ----------------------------
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_labels)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../backend/confusion_matrix.png")
plt.show()

# ----------------------------
# SAVE FINAL MODEL + LABELS
# ----------------------------
model.save("../backend/image_emotion_model.keras")

with open("../backend/image_label_classes.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("Training completed.")
print("Saved:")
print("../backend/image_emotion_model.keras")
print("../backend/best_emotion_model.keras (best val_accuracy checkpoint)")
print("../backend/image_label_classes.pkl")
print("../backend/confusion_matrix.png")