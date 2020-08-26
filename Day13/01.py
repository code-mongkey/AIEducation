import os
import sys
import cv2
import numpy as np

# define the input file
input_file = 'letter.data'

# define the visualization parameters
img_resize_factor = 12
start = 6
end = -1
height, width = 16, 8

# iterate until the user presses the Esc key
with open(input_file, 'r') as f:
    for line in f.readlines():
        # read the data
        data = np.array([255 * float(x) for x in line.split('\t')[start:end]])

        # reshape the data into a 2D image
        img = np.reshape(data, (height, width))

        # scale the image
        img_scaled = cv2.resize(img, None, fx=img_resize_factor, fy=img_resize_factor)

        # display the image
        cv2.imshow('image', img_scaled)

        # check if the user presed the Esc key
        c = cv2.waitKey()
        if c == 25:
            break

