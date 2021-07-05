import tensorflow as tf
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	labels="inferred", image_size=(64, 64), shuffle=True)
checkpoint_filepath = r"F:\Pycharm_projects\tensorflow_serving\models"

model = tf.keras.models.load_model(checkpoint_filepath)
model.summary()
"""who wrote this garbage?! """

image_path = r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test\PNEUMONIA\BACTERIA-1351146-0004.jpeg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=[128, 128])
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict_on_batch(input_arr)
print(np.argmax(predictions))
print(predictions)
