# Made by mithil salunkhe
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose, AvgPool2D, DepthwiseConv2D
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	labels="inferred", image_size=(64, 64), shuffle=True)
train = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train",
	labels="inferred", image_size=(64, 64), shuffle=True)
model = tf.keras.Sequential()

model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2DTranspose(kernel_size=2, filters=256, strides=1, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=512, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=512, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=256, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=256, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=128, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=128, padding="same"))
model.add(DepthwiseConv2D(kernel_size=2, strides=1))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(kernel_size=2, strides=1, filters=64, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=64, padding="same"))
model.add(DepthwiseConv2D(kernel_size=2, strides=1))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=32, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=32, padding="same"))

model.add(AvgPool2D(pool_size=2, strides=1, padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(kernel_size=2, strides=1, filters=16, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=16, padding="same"))
model.add(DepthwiseConv2D(kernel_size=2, strides=1))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=8, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=8, padding="same"))
model.add(DepthwiseConv2D(kernel_size=2, strides=1))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=4, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=4, padding="same"))

model.add(AvgPool2D(pool_size=2, strides=1, padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(tf.keras.layers.BatchNormalization())

model.add(Flatten())
model.add(Dense(1028, activation="relu"))

model.add(Dense(512, activation="relu"))

model.add(Dense(256, activation="relu"))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_filepath = r'F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\models '
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_accuracy',
	mode='max',
	save_best_only=True)
model.fit(train, validation_data=test, epochs=15, shuffle=True, batch_size=32, callbacks=model_checkpoint_callback)
model.load_weights(checkpoint_filepath)
model.summary()
tf.keras.utils.plot_model(model, to_file="dot_ig_file.png", show_shapes=True, show_layer_names=True, dpi=1200)
model.evaluate(test)
model.evaluate(train)
