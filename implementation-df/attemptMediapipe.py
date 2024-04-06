import cv2
import mediapipe as mp

# Setup Mediapipe solution
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize video capture for both cameras
cap_right = cv2.VideoCapture(4)
cap_left = cv2.VideoCapture(2)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap_right.isOpened() and cap_left.isOpened():
        # Read frames from both cameras
        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()

        # Convert frames to RGB
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Make frames unwritable to improve performance
        frame_right_rgb.flags.writeable = False
        frame_left_rgb.flags.writeable = False

        # Process each frame
        results_right = face_mesh.process(frame_right_rgb)
        results_left = face_mesh.process(frame_left_rgb)

        # Convert frames back to BGR for display
        frame_right = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)

        # Draw face mesh annotations on the frames
        for results, frame in [(results_left, frame_left), (results_right, frame_right)]:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

        # Display the frames
        cv2.imshow("Face Mesh Left", frame_left)
        cv2.imshow("Face Mesh Right", frame_right)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
