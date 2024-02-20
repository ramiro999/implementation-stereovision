import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

class DepthMap:
    def __init__(self, showImages):
        # Load the images
        self.imgL = cv.imread('left.jpg', 0)