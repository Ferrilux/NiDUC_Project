import cv2
import numpy as nm

image = cv2.imread('example.jpg',0)

ret, bw_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)


