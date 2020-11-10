import os
import cv2
import tensorflow as tf
import numpy as np
import tqdm


def make_data(image_path):
	global list_img
	global labels
	for i in tqdm(os.listdir(image_path)):
		list_img = []
		path = os.path.join(image_path, i)
		labels = []

		for ix in tqdm(os.listdir(path)):
			img = os.path.join(path, ix)
			img_decode = cv2.imread(img)
			list_img.append(img_decode)
	data_images = np.asarray(list_img).astype(np.float32)
	data_labels = np.asarray(labels).astype(np.float32)


make_data(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train")
