import cv2
import numpy as np
import calibration

# Open both cameras
cap_right = cv2.VideoCapture(2)
cap_left = cv2.VideoCapture(4)

# Stereo vision setup parameters
B = 11  # Distance between the cameras [cm]
f = 3.67  # Focal length [mm]

# Set parameters for the StereoSGBM algorithm
minDisparity = 0
numDisparities = 128  # Needs to be divisible by 16
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
                               speckleRange=speckleRange)

# Disparity map filtering
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

right_matcher = cv2.ximgproc.createRightMatcher(stereo)

def calculate_depth_map(filtered_disparity, f, B):
    # Convert to float32 disparity map
    disparity_map = np.float32(filtered_disparity) / 16.0
    # Avoid division by very small values
    disparity_map = np.maximum(disparity_map, 1)
    # Calculate the depth map
    depth_map = (f * B) / disparity_map
    # Normalize for visualization
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visual = np.uint8(depth_map_visual)
    return depth_map, depth_map_visual

while(cap_right.isOpened() and cap_left.isOpened()):
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    if not success_right or not success_left:
        break

    frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    frame_right_gray, frame_left_gray = calibration.undistortRectify(frame_right_gray, frame_left_gray)


    # Calculate the raw disparity map
    disparity_left = stereo.compute(frame_left_gray, frame_right_gray)
    disparity_right = right_matcher.compute(frame_right_gray, frame_left_gray)

    # Filtering the disparity map
    filtered_disparity = wls_filter.filter(disparity_left, frame_left_gray, None, disparity_right)
    filtered_disparity_visual = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    filtered_disparity_visual = np.uint8(filtered_disparity_visual)
    
    # Calculate and visualize the depth map
    _, depth_map_visual = calculate_depth_map(filtered_disparity, f, B)

    # Display the filtered and colored disparity map and the depth map
    cv2.imshow("Filtered Disparity", cv2.applyColorMap(filtered_disparity_visual, cv2.COLORMAP_JET))
    cv2.imshow("Depth Map", depth_map_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
