#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:11:30 2018

@author: vignesh
"""

import cv2
import numpy as np


def disp(I):
    I = np.uint8(I)
    cv2.imshow('image', I)
    cv2.waitKey(0)


def sobel_filtering(I, kernel_size):
    sobel_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = kernel_size)
    # disp(np.abs(sobel_x))
    sobel_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = kernel_size)
    # disp(np.abs(sobel_y))
    return sobel_x, sobel_y


def compute_gradient_and_direction(I):    
    # Apply Sobel filter in x and y directions
    sobel_x, sobel_y = sobel_filtering(I, 3)   
    # Compute Gradient magnitude
    grad_mag = np.sqrt(np.add(np.square(sobel_x), np.square(sobel_y)))    
    # Compute Gradient direction
    grad_dir = np.round(np.arctan2(sobel_y, sobel_x) * 180 / np.pi)
    grad_dir = grad_dir % 180
    # Quantize the directions to 0, 45, 90, 135
    angle_bins = np.array([0, 22.5, 67.5, 112.5, 157.5, 180.5])
    grad_dir = np.digitize(grad_dir, angle_bins)
    return grad_mag, grad_dir


def non_maximum_supression(grad_mag, grad_dir):
    # Get dimensions
    rows = grad_mag.shape[0]
    cols = grad_mag.shape[1]
    suppressed_I = grad_mag.copy()
    # Perform suppression by checking for local maxima
    for i in range(rows):
        for j in range(cols):
            direction = ((grad_dir[i][j] - 1) % 4) * 45
            if direction == 0:
                if grad_mag[i][max(j - 1, 0)] > grad_mag[i][j] or grad_mag[i][j] < grad_mag[i][min(j + 1, cols - 1)]:
                    suppressed_I[i][j] = 0
            elif direction == 45: 
                if grad_mag[max(i - 1, 0)][max(j - 1, 0)] > grad_mag[i][j] or grad_mag[i][j] < grad_mag[min(i + 1, rows - 1)][min(j + 1, cols - 1)]:
                    suppressed_I[i][j] = 0
            elif direction == 90:
                if grad_mag[min(i + 1, rows - 1)][j] > grad_mag[i][j] or grad_mag[i][j] < grad_mag[max(i - 1, 0)][j]:
                    suppressed_I[i][j] = 0
            elif direction == 135:
                if grad_mag[min(i + 1, rows - 1)][max(j - 1, 0)] > grad_mag[i][j] or grad_mag[i][j] < grad_mag[max(i - 1, 0)][min(j + 1, cols - 1)]:
                    suppressed_I[i][j] = 0
    return suppressed_I


def bfs(i, j, I, visited, rows, cols):
    visited[i][j] = 1
    queue = list([[i,j]])
    while queue:
        i, j = queue.pop(0)
        # Top-Left
        if i - 1 >= 0 and j - 1 >= 0 and I[i - 1][j - 1] > 0 and (not visited[i - 1][j - 1]):
            I[i - 1][j - 1] = 255
            queue.append([i - 1, j - 1])
            visited[i - 1][j - 1] = 1
        # Top
        if i - 1 >= 0 and I[i - 1][j] > 0 and (not visited[i - 1][j]):
            I[i - 1][j] = 255
            queue.append([i - 1, j])
            visited[i - 1][j] = 1
        # Top-Right
        if i - 1 >= 0 and j + 1 <= cols - 1 and I[i - 1][j + 1] > 0 and (not visited[i - 1][j + 1]):
            I[i - 1][j + 1] = 255
            queue.append([i - 1, j + 1])
            visited[i - 1][j + 1] = 1
        # Right
        if j + 1 <= cols - 1 and I[i][j + 1] > 0 and (not visited[i][j + 1]):
            I[i][j + 1] = 255
            queue.append([i, j + 1])
            visited[i][j + 1] = 1
        #  Bottom-Right
        if i + 1 <= rows - 1 and j + 1 <= cols - 1 and I[i + 1][j + 1] > 0 and (not visited[i + 1][j + 1]):
            I[i + 1][j + 1] = 255
            queue.append([i + 1, j + 1])
            visited[i + 1][j + 1] = 1
        # Bottom
        if i + 1 <= rows - 1 and I[i + 1][j] > 0 and (not visited[i + 1][j]):
            I[i + 1][j] = 255
            queue.append([i + 1, j])
            visited[i + 1][j] = 1
        # Bottom-Left
        if i + 1 <= rows - 1 and j - 1 >= 0 and I[i + 1][j - 1] > 0 and (not visited[i + 1][j - 1]):
            I[i + 1][j - 1] = 255
            queue.append([i + 1, j - 1])
            visited[i + 1][j - 1] = 1
        # Left
        if j - 1 >= 0 and I[i][j - 1] > 0 and (not visited[i][j - 1]):
            I[i][j - 1] = 255
            queue.append([i, j - 1])
            visited[i][j - 1] = 1
           

def hysteresis_thresholding(I, min_val, max_val):
    # Get dimensions
    rows = I.shape[0]
    cols = I.shape[1]
    thresholded_I = I.copy()
    strong_indexes = np.where(thresholded_I >= max_val) 
    weak_indexes = np.where(thresholded_I < min_val)
    thresholded_I[strong_indexes] = 255
    thresholded_I[weak_indexes] = 0
    strong_indexes = np.array(strong_indexes)
    # disp(I)
    visited = np.zeros((rows, cols))   
    # BFS to find connected points from the strong-edge points
    for k in range(strong_indexes.shape[1]):
        i = strong_indexes[0][k]
        j = strong_indexes[1][k]
        if not visited[i][j]:
            bfs(i, j, thresholded_I, visited, rows, cols)
    # Set unexplored edge-points to 0
    thresholded_I[np.where(thresholded_I != 255)] = 0
    return thresholded_I
    

def canny_edge_detector(I, min_val, max_val, kernel_size):
    # Get dimensions
    rows = I.shape[0]
    cols = I.shape[1]
    edge_I = np.zeros(I.shape, dtype = np.uint8)
    # Make BGR to GRAY
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur_I = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)    
    # Compute Gradient and its direction
    grad_mag, grad_dir = compute_gradient_and_direction(blur_I)
    # disp(grad_mag)
    # Non-maximum suppression (NMS)
    NMS_I = non_maximum_supression(grad_mag, grad_dir)
    # disp(NMS_I)
    # Hysteresis thresholding
    HT_I = hysteresis_thresholding(NMS_I, min_val, max_val)
    # disp(HT_I)
    edge_I = cv2.cvtColor(edge_I, cv2.COLOR_BGR2HSV) 
    grad_max = np.max(NMS_I)
    grad_min = np.min(NMS_I)
    for i in range(rows):
        for j in range(cols):
            if HT_I[i][j] == 255:
                direction = ((grad_dir[i][j] - 1) % 4) * 45
                value = int(255 * (grad_mag[i][j] - grad_min) / (grad_max - grad_min))
                if direction == 0:
                    edge_I[i][j] = [60, 255, value]
                elif direction == 45:
                    edge_I[i][j] = [30, 255, value]
                elif direction == 90:
                    edge_I[i][j] = [0, 255, value]
                else:
                    edge_I[i][j] = [120, 255, value]
    return cv2.cvtColor(edge_I, cv2.COLOR_HSV2BGR) 
    
    
img = cv2.imread('../Input_Images/building_large.jpg')
# disp(img)
edge_img = canny_edge_detector(img, 10, 20, 3)
disp(edge_img)
cv2.imwrite('../Output_Images/Canny/Best_Result/canny_output_best.png', edge_img)
cv2.destroyAllWindows()
        
