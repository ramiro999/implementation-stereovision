import cv2
import numpy as np
import time

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Open both cameras
cap_right = cv2.VideoCapture(2)
cap_left = cv2.VideoCapture(4)

# Stereo vision setup parameters
frame_rate = 120
B = 10
f = 3.67
alpha = 70

# Set parameters for the StereoSGBM algorithm
minDisparity = 0
numDisparities = 128
blockSize = 5
disp12MaxDiff = 1
uniquenessRatio = 3
speckleWindowSize = 10
speckleRange = 12

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange
                               )

# Creating variables for the filtering of the raw disparity map
right_matcher = cv2.ximgproc.createRightMatcher(stereo)

# Parameters for disparity map filtering
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


while(cap_right.isOpened() and cap_left.isOpened()):
    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    if not succes_right or not succes_left:
        break

    # Convert frames to grayscale (necessary for disparity calculation)
    frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    # Calculate the raw disparity map
    disparity_left = stereo.compute(frame_left_gray, frame_right_gray)
    disparity_left = np.float32(disparity_left) / 16.0  # Scale to proper values

    # Filtering the disparity map (this step enhances the quality of the disparity map)
    disparity_right = right_matcher.compute(frame_right_gray, frame_left_gray)
    disparity_right = np.float32(disparity_right) / 16.0  # Scale to proper values

    filtered_disparity = wls_filter.filter(disparity_left, frame_left_gray, None, disparity_right)
    filtered_disparity = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Now you can safely cast to uint8
    filtered_disparity = np.uint8(filtered_disparity)

    filtered_disparity = cv2.applyColorMap(filtered_disparity, cv2.COLORMAP_VIRIDIS)

    # Display the filtered disparity map
    cv2.imshow("Filtered Disparity", filtered_disparity)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()