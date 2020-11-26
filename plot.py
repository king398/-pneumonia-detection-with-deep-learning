import cv2

img = cv2.imread(
	r"F:\Pycharm_projects\pneumonia detection with deep learning\-pneumonia-detection-with-deep-learning\chest_xray\train\NORMAL\NORMAL-87870-0001.jpeg")
resize = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)

cv2.imshow("Resized image", resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("128imagenormal.jpg", resize)

print("losdld")