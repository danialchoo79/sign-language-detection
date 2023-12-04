import cv2
import mediapipe as mp
import os
import pandas as pd


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to detect hand landmarks and bounding box
def detect_hands(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    results = hands.process(frame_rgb)
    
    bounding_boxes = []
    features = (None, None, None, None)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
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
            width = x_max_norm - x_min_norm
            height = y_max_norm - y_min_norm
            features = (x_center, y_center, width, height)
    
    return bounding_boxes, features


def count_null_rows(df):
    is_null_df = df.isnull()
    null_rows = is_null_df.any(axis=1)
    return null_rows.sum()


in_root = 'clips/train'
out_root = 'sequences/train'
for label in ['z', 'o', 'j']:
    print(f'Directory size of {label}: {len(os.listdir(f"{in_root}/{label}/"))}')
    for filename in os.listdir(f'{in_root}/{label}/'):
        # Open the video file
        video_path = f'{in_root}/{label}/{filename}'
        cap = cv2.VideoCapture(video_path)
        sequence = []

        for _ in range(60): # Uniformly crop first 60 frames
            ret, frame = cap.read()
            
            if not ret:
                break
            
            hand_data, features = detect_hands(frame)
            sequence.append(features)
            
            for x_min, y_min, x_max, y_max in hand_data:
                # Draw bounding box on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Display the frame with bounding boxes
            cv2.imshow('Hand Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # print(len(sequence))
        df = pd.DataFrame(sequence, columns=['center_x', 'center_y', 'width', 'height'])
        if count_null_rows(df) > 2:
            print(f'{label}/{filename[:-4]}: {count_null_rows(df)}')
        else:
            df.to_csv(f'{out_root}/{label}/{filename[:-4]}.csv', index=False)
