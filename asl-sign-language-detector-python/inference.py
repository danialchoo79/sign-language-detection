import cv2
import mediapipe as mp
import numpy as np
from train_trail_classifier import get_trail_classifier
import torch


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Function to detect hand landmarks and bounding box
def detect_hands(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    results = hands.process(frame_rgb)
    
    bounding_boxes = []
    features = None
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks[:1]:
            # Calculate bounding box coordinates
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = float('inf'), float('inf'), float('-inf'), float('-inf')
            for landmark in landmarks.landmark:
                x_norm, y_norm, _ = landmark.x, landmark.y, landmark.z
                x_min_norm, y_min_norm = min(x_min_norm, x_norm), min(y_min_norm, y_norm)
                x_max_norm, y_max_norm = max(x_max_norm, x_norm), max(y_max_norm, y_norm)
                x, y = int(x_norm * frame.shape[1]), int(y_norm * frame.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
                
            bounding_boxes.append((x_min, y_min, x_max, y_max))

            x_center = (x_max_norm + x_min_norm) / 2
            y_center = (y_max_norm + y_min_norm) / 2
            features = (x_center, y_center)
    
    return bounding_boxes, features


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    norm_range = max(max_val - min_val, 0.25) # Don't want to normalize when the distance is too small (hand not moving)
    normalized_arr = (arr - min_val) / (norm_range) * 0.8 + 0.1
    return normalized_arr


def draw_image_from_coordinates(coords, width, height, line_thickness):
    # Create a blank black image
    image = np.zeros((height, width), dtype=np.uint8)

    if len(coords) >= 2:
        x_list, y_list = zip(*coords)

        # Normalize arrays to fill up image
        x = (normalize_array(np.array(x_list)) * width).astype(np.int32)
        y = (normalize_array(np.array(y_list)) * height).astype(np.int32)

        # Draw the lines on image    
        for i in range(len(x) - 1):
            cv2.line(image, (x[i], y[i]), (x[i+1], y[i+1]), color=255, thickness=line_thickness)

    return image


def predict_motion(trail_img, model):
    label_map = {0: 'j', 1: 'o', 2: 'z'}

    pred = None
    model.eval()
    with torch.no_grad():
        input_img = torch.tensor(trail_img).unsqueeze(0).to(DEVICE) / 255
        pred = torch.argmax(model(input_img).squeeze(0).to('cpu')).item()
    return label_map[pred]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_trail_classifier().to(DEVICE)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

coords_queue = []

WIDTH = 28
HEIGHT = 28
LINE_THICKNESS = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX 
ORG = (50, 50) 
FONTSCALE = 1
COLOR = (0, 255, 0) 


while True:
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)

    hand_data, features = detect_hands(frame)
    if features:
        coords_queue.append(features)
        if len(coords_queue) > 60:
            coords_queue.pop(0)

    for x_min, y_min, x_max, y_max in hand_data:
        # Draw bounding box on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    trail_img = draw_image_from_coordinates(coords_queue, WIDTH, HEIGHT, LINE_THICKNESS)

    motion_out = predict_motion(trail_img, model)
    if motion_out != 'o':
        cv2.putText(frame, f'Predicted class: {motion_out}', ORG, FONT, FONTSCALE, COLOR, LINE_THICKNESS, cv2.LINE_AA) 

    scale_factor = 10
    trail_img = cv2.resize(trail_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow('MediaPipe Hands', frame)
    cv2.imshow('Trail', trail_img)
    cv2.waitKey(1)     