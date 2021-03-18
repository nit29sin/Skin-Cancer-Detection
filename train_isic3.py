import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
#from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import cv2

train_examples = 20225
test_examples = 2555
validation_examples = 2551

img_height = img_width =224
batch_size1 = 32

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=["accuracy"]

)

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 15,
    zoom_range = (0.95, 0.95),
    horizontal_flip = True,
    vertical_flip = True,
    data_format = "channels_last",
    dtype = tf.float32,
)

validation_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size1,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size1,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size1,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

model.summary()
history = model.fit(
    train_gen,
    steps_per_epoch=train_examples//batch_size1,
    epochs=1,
    validation_data=validation_gen,
    validation_steps=validation_examples//batch_size1,
)

model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)

img = cv2.imread('data/test/malignant/ISIC_0014434_downsampled')
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])
classes = model.predict_classes(img)
print (classes)
