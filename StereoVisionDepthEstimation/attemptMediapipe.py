import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap_right = cv2.VideoCapture(4)
cap_left = cv2.VideoCapture(2)


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while (cap_right.isOpened() and cap_left.isOpened()):
        #ret, image = video.read()
        
        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()
        
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        frame_left.flags.writeable = False
        frame_right.flags.writeable = False
        results_left = face_mesh.process(frame_left)
        results_right = face_mesh.process(frame_right)
        # print(results)
        frame_left.flags.writeable = True
        frame_right.flags.writeable = True
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        if results_left.multi_face_landmarks:
            for face_landmarks in results_left.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=frame_left, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        if results_right.multi_face_landmarks:
            for face_landmarks in results_right.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=frame_right, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        cv2.imshow("Face Mesh Left", frame_left)
        cv2.imshow("Face Mesh Right", frame_right)
        k = cv2.waitKey(1)
        if k==ord('q'):
            break
    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #if results.multi_face_landmarks:
            #for face_landmarks in results.multi_face_landmarks:
             #   mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        #cv2.imshow("Face Mesh", image)
        #k = cv2.waitKey(1) # 1ms delay
        #if k==ord('q'):
         #   break
    #video.release()
    #cv2.destroyAllWindows()