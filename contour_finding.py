import cv2
import numpy as np


my_photo = cv2.imread('archive/cut3.png')
img_grey = cv2.cvtColor(my_photo,cv2.COLOR_BGR2GRAY)
filterd_image = cv2.GaussianBlur(img_grey, (5, 5), 0)

binary_adaptive = cv2.adaptiveThreshold(filterd_image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

contours, hierarchy = cv2.findContours(binary_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


img_contours = np.uint8(np.zeros((my_photo.shape[0],my_photo.shape[1])))
img_contours_max = np.uint8(np.zeros((my_photo.shape[0],my_photo.shape[1])))

min_area = 30
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

filtered_contours_sorted = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

max_cnt = filtered_contours_sorted[1]

cv2.drawContours(img_contours, filtered_contours, -1, (255,255,255), 1)
cv2.drawContours(img_contours_max, max_cnt, -1, (255,255,255), 1)

cv2.imshow('origin', my_photo)
cv2.imshow('res', img_contours)
cv2.imshow('max_cnt', img_contours_max)

cv2.waitKey()
cv2.destroyAllWindows()