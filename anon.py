#!/usr/bin/python3

import cv2

# Load the image
img = cv2.imread('poster_dataset/poster_gray.jpg')


# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_img.shape)

new_shape = [int(gray_img.shape[0]),int(gray_img.shape[1]/2)]

gray_img = gray_img[:new_shape[0],:new_shape[1]]

# # Save the grayscale image
cv2.imwrite('poster_dataset/poster_half_vert.jpg', gray_img)