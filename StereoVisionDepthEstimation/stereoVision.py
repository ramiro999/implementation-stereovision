import cv2
import mediapipe as mp
import numpy as np

# Import custom modules for stereo vision and calibration
import triangulation as tri
import calibration

# Setup for stereo vision and Mediapipe face mesh
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Stereo vision setup parameters
frame_rate, B, f, alpha = 120, 9, 8, 56.6

# Initialize face mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap_right.isOpened() and cap_left.isOpened():
        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()

        if not success_right or not success_left:
            print("Ignoring empty camera frame.")
            break

        # Calibration
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        # Convert frames to RGB
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        frames = {'right': frame_right_rgb, 'left': frame_left_rgb}
        centers = {}

        # Process frames with face mesh
        for side, frame_rgb in frames.items():
            results = face_mesh.process(frame_rgb)

            # Convert back to BGR for drawing and displaying
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Choose specific landmarks for depth estimation, e.g., tip of the nose
                    nose_tip_index = 1  # Example index for the tip of the nose
                    h, w, _ = frame_bgr.shape
                    nose_tip = np.multiply(
                        np.array((face_landmarks.landmark[nose_tip_index].x, face_landmarks.landmark[nose_tip_index].y)), [w, h]).astype(int)
                    centers[side] = nose_tip

                    # Draw the face mesh
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

            frames[side] = frame_bgr

        # Depth estimation if both sides have the landmark
        if 'right' in centers and 'left' in centers:
            depth = tri.find_depth(centers['right'], centers['left'], frame_right, frame_left, B, f, alpha)
            print("Depth: ", str(round(depth, 1)))

        # Display the frames
        cv2.imshow("Frame Right", frames['right'])
        cv2.imshow("Frame Left", frames['left'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

