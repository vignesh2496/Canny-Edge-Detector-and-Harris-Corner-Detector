    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:55:58 2018

@author: vignesh
"""

import cv2
import numpy as np


def disp(I):
    I = np.uint8(I)
    cv2.imshow('image', I)
    cv2.waitKey(0)
    
    
# 0 Mean 2D Gaussian
def gauss_2D(i, j, sigma):
    return np.exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
    
    
# 2D Gaussian kernel
def gaussian_kernel_2D(sigma, kernel_size):
    pad = int(kernel_size / 2)
    weights = [ [gauss_2D(i, j, sigma) for j in range(-pad, pad + 1)] for i in range(-pad, pad + 1) ]
    weights = np.array(weights)
    weights /= np.sum(weights)
    return weights 
    

def sobel_filtering(I, kernel_size):
    sobel_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = kernel_size)
    # disp(np.abs(sobel_x))
    sobel_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = kernel_size)
    # disp(np.abs(sobel_y))
    return sobel_x, sobel_y


def harris_corner_detector(I, nhood, k, kernel_size, gauss = False):
    # Make BGR to GRAY
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur_I = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0) 
    # Get dimensions
    rows = I.shape[0]
    cols = I.shape[1]
    # Apply Sobel filter in x and y directions
    sobel_x, sobel_y = sobel_filtering(blur_I, 3)
    IxIx = sobel_x * sobel_x
    IyIy = sobel_y * sobel_y
    IxIy = sobel_x * sobel_y
    if gauss:
        weights = gaussian_kernel_2D(1, nhood)
    else:
        weights = np.ones((nhood, nhood)) / (nhood * nhood)    
    pad = int(nhood / 2)
    corners = []
    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            xx = weights * IxIx[i - pad : i + pad + 1, j - pad : j + pad + 1]
            yy = weights * IyIy[i - pad : i + pad + 1, j - pad : j + pad + 1]
            xy = weights * IxIy[i - pad : i + pad + 1, j - pad : j + pad + 1]
            M_00 = xx.sum()
            M_01 = xy.sum()
            M_10 = xy.sum()
            M_11 = yy.sum()
            M = np.array([[M_00, M_01],[M_10, M_11]])
            det = np.linalg.det(M)
            trace = np.matrix.trace(M)
            R = det - k * (trace ** 2)
            corners.append([i, j, R])                      
    return np.array(corners)


def plot_corners_on_image(I, corners):
    corner_I = I.copy()
    for point in corners:
        i = int(point[0])
        j = int(point[1])
        corner_I[i][j] = [0, 0, 255]   
    return corner_I

img = cv2.imread('../Input_Images/building_small.jpg')
# disp(img)
corners = harris_corner_detector(img, 3, 0.04, 5, True)
corners = corners[np.where(corners[:, 2] > 0.01 * np.max(corners[:, 2]))]
print('Number of detected corner points : ', np.size(corners))
corner_img = plot_corners_on_image(img, corners)
disp(corner_img)
cv2.imwrite('../Output_Images/Harris/Best_Result/harris_output_best.png', corner_img)
cv2.destroyAllWindows()
