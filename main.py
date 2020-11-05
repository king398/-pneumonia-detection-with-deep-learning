#Made by mithil salunkhe
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose, AvgPool2D, DepthwiseConv2D
import numpy as np

train = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train",
	labels="inferred", image_size=(64, 64)
	, shuffle=True)

test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	labels="inferred", image_size=(64, 64), shuffle=True)


model = tf.keras.Sequential()

model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2DTranspose(kernel_size=2, filters=256, strides=1, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=256, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=256, padding="same"))


model.add(Conv2D(kernel_size=2, strides=1, filters=128, padding="same"))
model.add(Conv2D(kernel_size=2, strides=1, filters=128, padding="same"))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(kernel_size=2, strides=1, filters=64, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=64, padding="same"))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=32, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=32, padding="same"))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(kernel_size=2, strides=1, filters=16, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=16, padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=8, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=8, padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=4, padding="same"))

model.add(Conv2D(kernel_size=2, strides=1, filters=4, padding="same"))

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(tf.keras.layers.BatchNormalization())

model.add(Flatten())
model.add(Dense(1028, activation="relu"))

model.add(Dense(512, activation="relu"))

model.add(Dense(256, activation="relu"))

model.add(Dense(128, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(8, activation="relu"))

model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
checkpoint_filepath = r'F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\models '
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_accuracy',
	mode='max',
	save_best_only=True)
model.fit(train, validation_data=test, epochs=15, shuffle=True, batch_size=2, callbacks=model_checkpoint_callback)
model.load_weights(checkpoint_filepath)
model.summary()
tf.keras.utils.plot_model(model, to_file="dot_ig_file.png", show_shapes=True, show_layer_names=True, dpi=1200)
model.evaluate(test)
model.evaluate(train)
for x, y in test:
	print(x, y)
image_path = r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\images for showing\NORMAL-1283091-0001.jpeg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=[64, 64])
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = np.argmax(model.predict(input_arr))
if predictions == 0:
	print("The Patient is Not Infected With pneumonia")
elif predictions == 1:
	print("The Patient is Infected With pneumonia")
