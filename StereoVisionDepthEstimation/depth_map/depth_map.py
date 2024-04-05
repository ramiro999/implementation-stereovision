import numpy as np
import cv2
import calibration.calibration as calibration
import stereoMatching as sM

cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)

frame_rate = 120
B = 11
f = 3.67
alpha = 50

numDisparities = 16
blockSize = 15

while cap_right.isOpened() and cap_left.isOpened():
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    if not success_right or not success_left:
        break

    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

    frame_right_ = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

    center_right = 0
    center_left = 0

    depth_map = sM.depth_map(numDisparities, blockSize, center_right, center_left, frame_left, frame_right, B, f)

    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    # Normalizar el depth_map para visualización
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visual = np.uint8(depth_map_visual)

    #cv2.imshow("frame right", frame_right)
    #cv2.imshow("frame left", frame_left)
    cv2.imshow("depth map", depth_map_visual)  # Muestra el depth_map normalizado

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()