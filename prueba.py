import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

class DepthMap:
    def __init__(self, showImages):
        # Load the images
        root = os.getcwd()
        imgLeftPath = os.path.join(root, '/images/motorcycle/im0.png')
        imgRightPath = os.path.join(root, '/images/motorcycle/im1.png')
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show()

    def computeDepthMapBM(self):
        nDipsFactor = 6 #adjust this value to get better results
        stereo = cv.StereoBM.create(numDisparities=16*nDipsFactor, blockSize=21)
        disparity = stereo.compute(self.imgLeft, self.imgRight)
        plt. imshow(diparity, 'gray')
        plt.show()

    def computeDepthMapSGM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14 #adjust this (14 is good)
        num_disp = 16 * nDispFactor - min_disp

        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    blockSize=16, P1=8*3*window_size**2, 
                                    P2=32*3*window_size**2, 
                                    disp12MaxDiff=1, 
                                    uniquenessRatio=10, 
                                    speckleWindowSize=100, 
                                    speckleRange=32)
        
        # Compute the disparity map
        diparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0

        # Display the disparity map
        plt.imshow(diparity, 'gray')
        plt.colorbar()
        plt.show()


    def demoViewPics():
        dp = DepthMap(showImages=True)

    def demoStereoBM():
        dp = DepthMap(showImages=False)
        dp.computeDepthMapBM()

    def demoStereoSGBM():
        dp = DepthMap(showImages=False)
        dp.computeDepthMapSGBM()
    
    if __name__ == "__main__":
        demoViewPics()
        # demoStereoBM()
        # demoStereoSGBM()
        
