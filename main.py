import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten, LeakyReLU
from tensorflow.keras.utils import plot_model

train = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train",
	labels="inferred", image_size=(64, 64)
	, shuffle=True)
test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	labels="inferred", image_size=(64, 64), shuffle=True)
model = tf.keras.Sequential()
model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="tanh", padding="same"))
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=2, strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dense(32))
model.add(LeakyReLU())
model.add(Dense(16))
model.add(LeakyReLU())
model.add(Dense(2, activation="softmax"))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=test, batch_size=2, shuffle=True, epochs=1)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, dpi=1200)
model.evaluate(test)
print(type(test))

