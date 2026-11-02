import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ----------------------------
# SETTINGS
# ----------------------------
IMG_SIZE = 256

IMAGE_DIR = "Shenzhen/img"
MASK_DIR = "Shenzhen/mask"

# ----------------------------
# LOAD DATA
# ----------------------------
images = []
masks = []

image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

for img_name, mask_name in zip(image_files, mask_files):

    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    mask = (mask > 0.5).astype(np.float32)

    images.append(img)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)

images = np.expand_dims(images, axis=-1)
masks = np.expand_dims(masks, axis=-1)

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# U-NET MODEL
# ----------------------------
inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1))

# Encoder
c1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
c1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
p1 = tf.keras.layers.MaxPooling2D()(c1)

c2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
c2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
p2 = tf.keras.layers.MaxPooling2D()(c2)

c3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
c3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c3)

# Decoder
u1 = tf.keras.layers.UpSampling2D()(c3)
u1 = tf.keras.layers.Concatenate()([u1, c2])

c4 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(u1)
c4 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c4)

u2 = tf.keras.layers.UpSampling2D()(c4)
u2 = tf.keras.layers.Concatenate()([u2, c1])

c5 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
c5 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c5)

outputs = tf.keras.layers.Conv2D(
    1,
    1,
    activation="sigmoid"
)(c5)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("lung_segmentation_unet.h5")

print("\nTraining Complete!")
print("Model saved as lung_segmentation_unet.h5")