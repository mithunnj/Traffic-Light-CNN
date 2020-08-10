import os
import cv2
import numpy as np

filepath = os.path.dirname(os.getcwd()) + "/cropped_imgs/21382/21382_6.jpg"
img = cv2.imread(filepath)
resized_img = cv2.resize(img, (28,28))

numpy_horizontal_concat = np.concatenate((img, resized_img), axis=1)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
cv2.waitKey(0)


