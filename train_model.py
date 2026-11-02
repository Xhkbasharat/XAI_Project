import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# 🔥 Data Augmentation (Advanced)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 🔥 Transfer Learning (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Custom layers (Advanced)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)   # 🔥 prevents overfitting
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔥 Early Stopping (Advanced)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Train model
model.fit(train, validation_data=val, epochs=10, callbacks=[early_stop])

# Save model
model.save("model.h5")

print("✅ Model training completed and saved as model.h5")