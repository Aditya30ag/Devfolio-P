import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hand_detection = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

known_face_encodings = []
known_face_names = []

# Blink detection parameters
eye_aspect_ratio_threshold = 0.2  # Adjust based on testing
consecutive_frames_closed = {}

def compute_face_encoding(face_image):
    if face_image.size == 0:
        return None
    return np.random.rand(128)

def load_and_encode_known_faces():
    image1 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
    if image1 is not None:
        encoding1 = compute_face_encoding(image1)
        if encoding1 is not None:
            known_face_encodings.append(encoding1)
            known_face_names.append("Person 1")
            print("Image 1 found.")
        else:
            print("Encoding for Image 1 failed.")
    else:
        print("Image 1 not found.")

    image2 = cv2.imread(r"C:\Users\adity\Pictures\Screenshots\Screenshot 2024-10-27 140417.png")
    if image2 is not None:
        encoding2 = compute_face_encoding(image2)
        if encoding2 is not None:
            known_face_encodings.append(encoding2)
            known_face_names.append("Person 2")
            print("Image 2 found.")
        else:
            print("Encoding for Image 2 failed.")
    else:
        print("Image 2 not found.")

load_and_encode_known_faces()

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices):
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    return (A + B) / (2.0 * C)

# Indices of landmarks around the eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

video_capture = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=2, refine_landmarks=True) as face_mesh, \
     mp_hand_detection.Hands(min_detection_confidence=0.5) as hand_detection:

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        face_results = face_detection.process(rgb_frame)
        face_mesh_results = face_mesh.process(rgb_frame)

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                       int(bboxC.width * w), int(bboxC.height * h))

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                
                face_image = rgb_frame[y:y + height, x:x + width]
                face_encoding = compute_face_encoding(face_image)
                if face_encoding is None:
                    continue

                matches = [np.linalg.norm(face_encoding - known_encoding) < 0.6 for known_encoding in known_face_encodings]
                name = "Unknown"
                if any(matches):
                    best_match_index = np.argmin([np.linalg.norm(face_encoding - known_encoding) for known_encoding in known_face_encodings])
                    name = known_face_names[best_match_index]
                    
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                mask[y:y + height, x:x + width] = 0

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        # Draw rectangles around the eyes
                        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                            eye_points = [(int(face_landmarks.landmark[i].x * w),
                                           int(face_landmarks.landmark[i].y * h)) for i in eye_indices]
                            x_min = min([p[0] for p in eye_points])
                            y_min = min([p[1] for p in eye_points])
                            x_max = max([p[0] for p in eye_points])
                            y_max = max([p[1] for p in eye_points])
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)

                        # Calculate EAR for both eyes
                        left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                        right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0

                        # Blink detection based on EAR
                        if name:
                            if avg_ear < eye_aspect_ratio_threshold:
                                consecutive_frames_closed[name] = consecutive_frames_closed.get(name, 0) + 1
                            else:
                                consecutive_frames_closed[name] = 0

                            if consecutive_frames_closed[name] == 1:  # Blink detected
                                cv2.putText(frame, f"{name} blinked!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Process hands and background color as in your original code
        hand_results = hand_detection.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, y), max(y_max, y)

                cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 255, 0), 2)
                cv2.putText(frame, "Hand", (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                mask[y_min:y_max, x_min:x_max] = 0

        # Display final frame with blink and hand detection
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
