# ============================================================
# FAST CPU CNN - EMOTION DETECTION
# ------------------------------------------------------------
# Training:
#   1,000 images PER CLASS from the training dataset
#
# Validation:
#   ALL validation images
#
# Testing:
#   ALL test images
#
# Designed for CPU-only training in VS Code
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import glob
import random
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    SeparableConv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_DIR = "emotion detection dataset main"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = 48

# ------------------------------------------------------------
# IMPORTANT:
# Only 1,000 training images per class
# ------------------------------------------------------------
IMAGES_PER_CLASS = 3000

# CPU-friendly batch size
BATCH_SIZE = 256

# Maximum epochs
# EarlyStopping will normally stop before this
EPOCHS = 50

SEED = 42

L2_REG = 1e-4

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 4. CPU CONFIGURATION
# ============================================================

CPU_COUNT = os.cpu_count()

print("=" * 60)
print("CPU CONFIGURATION")
print("=" * 60)

print("CPU cores detected:", CPU_COUNT)

# Use available CPU threads
try:
    tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(CPU_COUNT)
except RuntimeError:
    # TensorFlow may already have initialized the threading system
    pass

print()


# ============================================================
# 5. CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VALIDATION_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# 6. CLASS NAMES
# ============================================================

class_names = sorted([
    name
    for name in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, name))
])

num_classes = len(class_names)

class_to_idx = {
    name: i
    for i, name in enumerate(class_names)
}

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print()


# ============================================================
# 7. GET FILES FOR EACH CLASS
# ============================================================

def get_class_files(directory, class_name):

    class_dir = os.path.join(directory, class_name)

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
        "*.BMP"
    ]

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(class_dir, ext)
            )
        )

    return files


# ============================================================
# 8. SELECT TRAINING IMAGES
# ============================================================
#
# EXACTLY 1,000 images per class
#
# If a class contains fewer than 1,000 images,
# all available images will be used.
# ============================================================

print("=" * 60)
print("SELECTING TRAINING DATA")
print("=" * 60)

train_files = []
train_labels = []

for class_name in class_names:

    class_idx = class_to_idx[class_name]

    files = get_class_files(
        TRAIN_DIR,
        class_name
    )

    print(
        f"{class_name}: "
        f"{len(files)} images available"
    )

    # Shuffle files
    random.shuffle(files)

    # Take maximum 1,000
    selected_files = files[:IMAGES_PER_CLASS]

    print(
        f"         -> "
        f"{len(selected_files)} images selected"
    )

    train_files.extend(selected_files)

    train_labels.extend(
        [class_idx] * len(selected_files)
    )

print()


# ============================================================
# 9. LOAD ALL VALIDATION IMAGES
# ============================================================

val_files = []
val_labels = []

for class_name in class_names:

    class_idx = class_to_idx[class_name]

    files = get_class_files(
        VALIDATION_DIR,
        class_name
    )

    val_files.extend(files)

    val_labels.extend(
        [class_idx] * len(files)
    )


# ============================================================
# 10. LOAD ALL TEST IMAGES
# ============================================================

test_files = []
test_labels = []

for class_name in class_names:

    class_idx = class_to_idx[class_name]

    files = get_class_files(
        TEST_DIR,
        class_name
    )

    test_files.extend(files)

    test_labels.extend(
        [class_idx] * len(files)
    )


# ============================================================
# 11. SHUFFLE TRAINING DATA
# ============================================================

combined = list(
    zip(train_files, train_labels)
)

random.shuffle(combined)

train_files, train_labels = zip(*combined)

train_files = list(train_files)
train_labels = list(train_labels)


# ============================================================
# 12. DATASET SUMMARY
# ============================================================

print("=" * 60)
print("FINAL DATASET SIZE")
print("=" * 60)

print(
    "Training images:",
    len(train_files)
)

print(
    "Validation images:",
    len(val_files)
)

print(
    "Testing images:",
    len(test_files)
)

print()

print(
    "Training images per class:",
    IMAGES_PER_CLASS
)

print()


# ============================================================
# 13. CLASS DISTRIBUTION
# ============================================================

print("=" * 60)
print("TRAINING DISTRIBUTION")
print("=" * 60)

for class_name in class_names:

    class_idx = class_to_idx[class_name]

    count = train_labels.count(class_idx)

    print(
        f"{class_name}: {count}"
    )

print()


# ============================================================
# 14. CLASS WEIGHTS
# ============================================================
#
# Since we're intentionally taking the same number of
# images per class, the class weights should normally be
# approximately equal.
#
# We still calculate them to keep the pipeline robust.
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight_dict = dict(
    enumerate(class_weights_array)
)

class_weights_tensor = tf.constant(
    [
        class_weight_dict[i]
        for i in range(num_classes)
    ],
    dtype=tf.float32
)

print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

print(class_weight_dict)

print()


# ============================================================
# 15. IMAGE LOADING + PREPROCESSING
# ============================================================

def load_and_preprocess(filepath, label):

    # Read image
    img = tf.io.read_file(filepath)

    # Decode as grayscale
    img = tf.io.decode_image(
        img,
        channels=1,
        expand_animations=False
    )

    # Restore shape information
    img.set_shape(
        [None, None, 1]
    )

    # Resize to 48x48
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE]
    )

    # Convert to float
    img = tf.cast(
        img,
        tf.float32
    )

    # Normalize 0-255 -> 0-1
    img = img / 255.0

    # One-hot label
    label = tf.one_hot(
        label,
        num_classes
    )

    return img, label


# ============================================================
# 16. DATA AUGMENTATION
# ============================================================

def augment_fn(img, label):

    # Horizontal flip
    img = tf.image.random_flip_left_right(
        img
    )

    # Small brightness variation
    img = tf.image.random_brightness(
        img,
        max_delta=0.10
    )

    # Small contrast variation
    img = tf.image.random_contrast(
        img,
        lower=0.90,
        upper=1.10
    )

    # Small translation/crop
    img = tf.image.resize_with_crop_or_pad(
        img,
        IMG_SIZE + 4,
        IMG_SIZE + 4
    )

    img = tf.image.random_crop(
        img,
        [
            IMG_SIZE,
            IMG_SIZE,
            1
        ]
    )

    # Keep values between 0 and 1
    img = tf.clip_by_value(
        img,
        0.0,
        1.0
    )

    return img, label


# ============================================================
# 17. ADD SAMPLE WEIGHTS
# ============================================================

def add_sample_weight(img, label):

    class_idx = tf.argmax(
        label,
        output_type=tf.int32
    )

    weight = tf.gather(
        class_weights_tensor,
        class_idx
    )

    return img, label, weight


# ============================================================
# 18. CREATE TF.DATA DATASET
# ============================================================

def make_dataset(
    filepaths,
    labels,
    augment=False,
    shuffle=False,
    weighted=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            filepaths,
            labels
        )
    )

    # Load and preprocess
    ds = ds.map(
        load_and_preprocess,
        num_parallel_calls=AUTOTUNE
    )

    # Cache decoded/resized images in RAM
    #
    # This means subsequent epochs don't need to repeatedly
    # read and decode the same JPEG files.
    ds = ds.cache()

    # Shuffle only training dataset
    if shuffle:

        ds = ds.shuffle(
            buffer_size=len(filepaths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # Augmentation happens AFTER cache
    #
    # Therefore the original image is cached, but a new
    # random augmentation is generated every epoch.
    if augment:

        ds = ds.map(
            augment_fn,
            num_parallel_calls=AUTOTUNE
        )

    # Add sample weights
    if weighted:

        ds = ds.map(
            add_sample_weight,
            num_parallel_calls=AUTOTUNE
        )

    # Batch
    ds = ds.batch(
        BATCH_SIZE
    )

    # Prefetch
    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds


# ============================================================
# 19. CREATE DATASETS
# ============================================================

print("=" * 60)
print("CREATING TF.DATA PIPELINES")
print("=" * 60)

print("Preparing training dataset...")

train_ds = make_dataset(
    train_files,
    train_labels,
    augment=True,
    shuffle=True,
    weighted=True
)

print("Preparing validation dataset...")

val_ds = make_dataset(
    val_files,
    val_labels,
    augment=False,
    shuffle=False,
    weighted=False
)

print("Preparing test dataset...")

test_ds = make_dataset(
    test_files,
    test_labels,
    augment=False,
    shuffle=False,
    weighted=False
)

print("Dataset pipelines ready.")
print()


# ============================================================
# 20. BUILD CNN MODEL
# ============================================================

print("=" * 60)
print("BUILDING CNN MODEL")
print("=" * 60)


def conv_block(
    filters,
    kernel_size,
    dropout_rate,
    separable=False
):

    if separable:

        conv = SeparableConv2D(
            filters,
            kernel_size,
            padding="same",
            depthwise_regularizer=l2(L2_REG),
            pointwise_regularizer=l2(L2_REG)
        )

    else:

        conv = Conv2D(
            filters,
            kernel_size,
            padding="same",
            kernel_regularizer=l2(L2_REG)
        )

    return [
        conv,
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(
            pool_size=(2, 2)
        ),
        Dropout(dropout_rate)
    ]


model = Sequential([

    Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            1
        )
    ),

    # ========================================================
    # BLOCK 1
    # ========================================================

    *conv_block(
        32,
        (3, 3),
        0.20,
        separable=False
    ),

    # ========================================================
    # BLOCK 2
    # ========================================================

    *conv_block(
        64,
        (3, 3),
        0.25,
        separable=True
    ),

    # ========================================================
    # BLOCK 3
    # ========================================================

    *conv_block(
        128,
        (3, 3),
        0.30,
        separable=True
    ),

    # ========================================================
    # BLOCK 4
    # ========================================================

    *conv_block(
        256,
        (3, 3),
        0.35,
        separable=True
    ),

    # ========================================================
    # GLOBAL FEATURE EXTRACTION
    # ========================================================

    GlobalAveragePooling2D(),

    # ========================================================
    # FULLY CONNECTED LAYER
    # ========================================================

    Dense(
        128,
        kernel_regularizer=l2(L2_REG)
    ),

    BatchNormalization(),

    Activation("relu"),

    Dropout(0.5),

    # ========================================================
    # OUTPUT
    # ========================================================

    Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# 21. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=[
        "accuracy"
    ],

    # Reduces Python overhead
    steps_per_execution=8

    # NOTE:
    # jit_compile=True is intentionally NOT used.
    #
    # On CPU-only systems XLA compilation can sometimes
    # increase startup time or provide little/no benefit.
)


# ============================================================
# 22. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 23. CREATE BACKEND DIRECTORY
# ============================================================

BACKEND_DIR = "../backend"

os.makedirs(
    BACKEND_DIR,
    exist_ok=True
)


# ============================================================
# 24. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=7,

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

    os.path.join(
        BACKEND_DIR,
        "best_emotion_model.keras"
    ),

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


# ============================================================
# 25. TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)

print(
    f"Training images: {len(train_files)}"
)

print(
    f"Images per class: {IMAGES_PER_CLASS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Maximum epochs: {EPOCHS}"
)

print(
    "CPU cores:",
    CPU_COUNT
)

print("=" * 60)
print()


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)


# ============================================================
# 26. EVALUATE ON COMPLETE TEST SET
# ============================================================

print()
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print()
print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print()
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred_probs = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

y_true = np.array(
    test_labels
)


# ============================================================
# 28. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# 29. CONFUSION MATRIX
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "True"
)

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()


confusion_matrix_path = os.path.join(
    BACKEND_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150
)

plt.show()


# ============================================================
# 30. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    BACKEND_DIR,
    "image_emotion_model.keras"
)

model.save(
    final_model_path
)


# ============================================================
# 31. SAVE CLASS LABELS
# ============================================================

label_path = os.path.join(
    BACKEND_DIR,
    "image_label_classes.pkl"
)

with open(
    label_path,
    "wb"
) as f:

    pickle.dump(
        class_names,
        f
    )


# ============================================================
# 32. TRAINING SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print()
print("Saved files:")

print(
    final_model_path
)

print(
    os.path.join(
        BACKEND_DIR,
        "best_emotion_model.keras"
    )
)

print(
    label_path
)

print(
    confusion_matrix_path
)

print()
print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 60)