"""
This code aims to learn the landmark features and labels and store it in a pickle file as data and labels

- Danial C.

"""

import os
import csv
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'./resized_images'
CSV_FILE = 'train.csv'
PICKLE_FILE = 'data.pickle'

data = []
labels = []
max_length = 42

# First loop to determine the maximum length and for padding calculation
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                max_length = max(max_length, len(hand_landmarks.landmark) * 2)  # x and y for each landmark

# Initialize CSV file with headers
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ['filename'] + [f'x{i}' for i in range(max_length//2)] + [f'y{i}' for i in range(max_length//2)]
    writer.writerow(headers)

# Second loop to actually process and pad data
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y])
            print(data_aux)

            # Write to CSV
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                filename = os.path.join(dir_, img_path)
                row = [filename] + data_aux
                writer.writerow(row)
            
            # Add data to the lists for pickle file
            data.append(data_aux)
            labels.append(dir_)

# Save data and labels in a pickle file
with open(PICKLE_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(data[0])
print(labels[0])

# THIS COMMENTED OUT CODE IS THE BASE CODE FOR REPRODUCTION
# DANIAL C.

# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = r'./resized_images'

# data = []
# labels = []
# max_length = 84

# # First loop to determine the maximum length
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 max_length = max(max_length, len(hand_landmarks.landmark) * 2)  # x and y for each landmark

# # Second loop to actually process and pad data
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for landmark in hand_landmarks.landmark:
#                     data_aux.extend([landmark.x, landmark.y])

#             # Pad data_aux if it's shorter than max_length
#             data_aux.extend([0] * (max_length - len(data_aux)))

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()