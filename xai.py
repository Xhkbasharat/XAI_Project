import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("model.h5")

# 👉 PUT YOUR IMAGE PATH HERE (IMPORTANT)
img_path = r"C:\Users\HP\OneDrive\Desktop\XAI_Project\dataset\NORMAL\IM-0122-0001.jpeg"

# Read image
img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found. Check path!")
    exit()

# Preprocess image
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Get last convolution layer (safe way)
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

# Grad-CAM model
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

# Fix dimension issue
pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

conv_outputs = conv_outputs[0]

# Generate heatmap
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Convert heatmap to color
heatmap = cv2.resize(heatmap,(224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Original image
original = cv2.imread(img_path)
original = cv2.resize(original, (224, 224))

# Overlay heatmap
superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Show results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("XAI Heatmap")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()