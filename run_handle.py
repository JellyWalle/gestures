import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]
    
      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())
    
      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN
    
      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

# Create an HandLandmarker object.
Base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
RESULT = None

# Create a hand landmarker instance with the live stream mode:
def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result
    
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions(base_options=Base_options,
                                                                    num_hands=2,
                                                                    running_mode=VisionRunningMode.LIVE_STREAM,
                                                                    result_callback=get_result)
 
detector = HandLandmarker.create_from_options(HandLandmarkerOptions)

cap = cv2.VideoCapture(0)
timestamp=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    w, h, c = frame.shape
    rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detector.detect_async(mp_image, timestamp)
    if RESULT is not None:
        frame = draw_landmarks_on_image(frame,RESULT)
        
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    timestamp+=1
    
