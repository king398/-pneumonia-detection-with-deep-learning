# Made by mithil salunkhe
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Conv2DTranspose, AvgPool2D, LeakyReLU, \
	BatchNormalization
from tensorflow.keras import layers
import numpy as np
import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	image_size=(64, 64))
train = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train",
	image_size=(64, 64), shuffle=True)
model = tf.keras.Sequential()

layers.experimental.preprocessing.Rescaling(1. / 255)
layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
layers.experimental.preprocessing.RandomRotation(0.2),
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=1, strides=1))

model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=256, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=128, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=32, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(10, activation="softmax"))
tf.keras.optimizers.Adam(
	learning_rate=0.0001, )
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
checkpoint_filepath = r'F:/Pycharm_projects\pneumonia detection with deep learning/-pneumonia-detection-with-deep-learning/models'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train, validation_data=test, epochs=15, batch_size=32,
          callbacks=[model_checkpoint_callback, tensorboard_callback])
model.load(checkpoint_filepath)
model.summary()
tf.keras.utils.plot_model(model, to_file="dot_ig_file.png", show_shapes=True, show_layer_names=True, dpi=1200)
model.evaluate(test)
model.evaluate(train)
image_path = r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\images for showing\person1946_bacteria_4875.jpeg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=[64, 64])
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(np.argmax(predictions))
model.save(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\model_final",
	overwrite=True, include_optimizer=True)
