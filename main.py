import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

model = tf.keras.Sequential()
model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same"))
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(AveragePooling2D(pool_size=2, strides=1, padding="same"))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same"))
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(AveragePooling2D(pool_size=2, strides=1, padding="same"))
model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.build(input_shape=[32, 32, 32, 1])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, dpi=1200)
