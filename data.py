import tensorflow as tf
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
test = tf.keras.preprocessing.image_dataset_from_directory(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\test",
	labels="inferred", image_size=(64, 64), shuffle=True)
checkpoint_filepath = r'F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\models'

model = tf.keras.models.load_model(checkpoint_filepath)
model.summary()
model.evaluate(test)

image_path = r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\images for showing\NORMAL2-IM-1431-0001.jpeg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=[64, 64])
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(np.argmax(predictions))
