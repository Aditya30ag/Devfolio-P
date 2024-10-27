# import cv2
# import mediapipe as mp
# import numpy as np

# mp_face_detection = mp.solutions.face_detection
# mp_hand_detection = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# known_face_encodings = []
# known_face_names = []

# def compute_face_encoding(face_image):
#     # Placeholder for a real face encoding function
#     return np.random.rand(128)

# def load_and_encode_known_faces():
#     # Load the first image
#     image1 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
#     if image1 is not None:
#         encoding1 = compute_face_encoding(image1)
#         known_face_encodings.append(encoding1)
#         known_face_names.append("Person 1")
#         print("Image 1 found.")
#     else:
#         print("Image 1 not found.")

#     # Load the second image
#     image2 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
#     if image2 is not None:
#         encoding2 = compute_face_encoding(image2)
#         known_face_encodings.append(encoding2)
#         known_face_names.append("Person 2")
#     else:
#         print("Image 2 not found.")

# load_and_encode_known_faces()

# video_capture = cv2.VideoCapture(0)

# with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
#      mp_hand_detection.Hands(min_detection_confidence=0.5) as hand_detection:
    
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Failed to capture video")
#             break
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, _ = frame.shape  # Get the dimensions after color conversion

#         face_results = face_detection.process(rgb_frame)
#         face_names = []
        
#         if face_results.detections:
#             for detection in face_results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
#                                         int(bboxC.width * w), int(bboxC.height * h))

#                 cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

#                 face_image = rgb_frame[y:y + height, x:x + width]
#                 face_encoding = compute_face_encoding(face_image)

#                 # Check if there's a match
#                 matches = [np.linalg.norm(face_encoding - known_encoding) < 0.6 for known_encoding in known_face_encodings]
#                 name = "Unknown"
#                 if any(matches):
#                     best_match_index = np.argmin([np.linalg.norm(face_encoding - known_encoding) for known_encoding in known_face_encodings])
#                     name = known_face_names[best_match_index]
                    
#                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
#         hand_results = hand_detection.process(rgb_frame)
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 x_min, y_min = w, h
#                 x_max, y_max = 0, 0
#                 for landmark in hand_landmarks.landmark:
#                     x = int(landmark.x * w)
#                     y = int(landmark.y * h)
#                     x_min, y_min = min(x_min, x), min(y_min, y)
#                     x_max, y_max = max(x_max, x), max(y_max, y)
                    
#                 cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 255, 0), 2)
#                 cv2.putText(frame, "Hand", (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

#         cv2.imshow("Video", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# video_capture.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_hand_detection = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

known_face_encodings = []
known_face_names = []

def compute_face_encoding(face_image):
    # Placeholder for a real face encoding function
    return np.random.rand(128)

def load_and_encode_known_faces():
    # Load the first image
    image1 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
    if image1 is not None:
        encoding1 = compute_face_encoding(image1)
        known_face_encodings.append(encoding1)
        known_face_names.append("Person 1")
        print("Image 1 found.")
    else:
        print("Image 1 not found.")

    # Load the second image
    image2 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
    if image2 is not None:
        encoding2 = compute_face_encoding(image2)
        known_face_encodings.append(encoding2)
        known_face_names.append("Person 2")
    else:
        print("Image 2 not found.")

load_and_encode_known_faces()

video_capture = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hand_detection.Hands(min_detection_confidence=0.5) as hand_detection:
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape  # Get the dimensions after color conversion

        face_results = face_detection.process(rgb_frame)
        face_names = []
        
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                        int(bboxC.width * w), int(bboxC.height * h))

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

                face_image = rgb_frame[y:y + height, x:x + width]
                face_encoding = compute_face_encoding(face_image)

                # Check if there's a match
                matches = [np.linalg.norm(face_encoding - known_encoding) < 0.6 for known_encoding in known_face_encodings]
                name = "Unknown"
                if any(matches):
                    best_match_index = np.argmin([np.linalg.norm(face_encoding - known_encoding) for known_encoding in known_face_encodings])
                    name = known_face_names[best_match_index]
                    
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                # Eye detection
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)  # Draw rectangle around eyes
        
        hand_results = hand_detection.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                    
                cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 255, 0), 2)
                cv2.putText(frame, "Hand", (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
