import numpy as np
import cv2
import os

def hough(thr):
    dir = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread('images/sudoku.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, thr)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough(thr, minLineLength, maxLineGap):
    img = cv2.imread('images/sudoku.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, thr, minLineLength, maxLineGap)

    for line in lines:
        x1, y1, x2, y2 = line[0]        
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hough(200, 100, 10)