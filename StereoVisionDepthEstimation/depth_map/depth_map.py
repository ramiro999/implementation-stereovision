import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import calibration.calibration as calibration
import stereoMatching as sM

# Open both cameras
cap_right = cv2.VideoCapture(1)                    
cap_left =  cv2.VideoCapture(0)

# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 11               #Distance between the cameras [cm]
f = 3.67             #Camera lense's focal length [mm]
alpha = 50       #Camera field of view in the horisontal plane [degrees]

# Stereo Matching parameters
numDisparities = 16
blockSize = 15


while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    ################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    ########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:                    
        break

    else:

        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to BGR
        frame_right_ = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


        ################## CALCULATING DEPTH_MAP #########################################################

        center_right = 0
        center_left = 0

        # Función que calcula el depth_map
        depth_map =  sM.depth_map(numDisparities, blockSize, center_right, center_left, frame_left, frame_right, B, f)

        # Show the frames
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()