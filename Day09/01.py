#-*- coding:utf-8 -*-
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import os

print(os.path.dirname( os.path.abspath( __file__ )))
print('./images')

print (os.getcwd()) #현재 디렉토리의
print (os.path.realpath(__file__))#파일
print (os.path.dirname(os.path.realpath(__file__)) )#파일이 위치한 디렉토리

dir = os.path.dirname(os.path.realpath(__file__))

color_img1 = cv2.imread(dir + '/images/rose.jpg',cv2.IMREAD_COLOR)
color_img2 = cv2.imread(dir + '/images/sunflower.jpg',cv2.IMREAD_COLOR)

color_img2 = cv2.resize(color_img2, dsize=(600,600), interpolation=cv2.INTER_AREA)

add_img = cv2.add(color_img1, color_img2)

img1 = cv2.cvtColor(color_img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(color_img2, cv2.COLOR_RGB2GRAY)

print(img1.shape)
print(img1.size)
print(img1.dtype)

hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])

plt.subplot(221),plt.imshow(img1,'gray'),plt.title('Red Line')
plt.subplot(222),plt.imshow(img2,'gray'),plt.title('Green Line')
plt.subplot(223),plt.plot(hist1,color='r'),plt.plot(hist2,color='g')
plt.subplot(224),plt.imshow(add_img, 'g'),plt.title('gg')




plt.xlim([0,256])

plt.show()


