# Made by mithil salunkhe
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose, AvgPool2D, LeakyReLU, \
	BatchNormalization

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
model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(Conv2D(filters=256, kernel_size=2, strides=1, activation="relu"))
model.add(Drop)
model.add(Conv2D(filters=128, kernel_size=2, strides=1))
model.add(Conv2D(filters=128, kernel_size=2, strides=1))
model.add(Conv2D(filters=64, kernel_size=2, strides=1))
model.add(Conv2D(filters=64, kernel_size=2, strides=1))

model.add(Flatten())
model.add(Dense(512, activation="relu"))

model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(2, activation="relu"))
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

checkpoint_filepath = r'F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\models '
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_accuracy', )

model.fit(train, validation_data=test, epochs=15, shuffle=True, batch_size=16, callbacks=model_checkpoint_callback, )
model.load_weights(checkpoint_filepath)
model.summary()
tf.keras.utils.plot_model(model, to_file="dot_ig_file.png", show_shapes=True, show_layer_names=True, dpi=1200)
model.evaluate(test)
model.evaluate(train)
image_path = r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\images for showing\NORMAL-1283091-0001.jpeg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=[64, 64])
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(np.argmax(predictions))
