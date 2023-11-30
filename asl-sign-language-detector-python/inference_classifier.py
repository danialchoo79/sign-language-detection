""" 
This code aims to infer from from the trained data, and predict the labels. 

It also includes UI, Gamification, Music and more.

- Danial C.

"""

import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# --------------ASL Image Setup--------------------------------------------
asl_image_path = 'asl-sign-language-detector-python/asl.png'
asl_image = cv2.imread(asl_image_path)

# Customizable dimensions for ASL image
asl_width = 600
asl_height = 500
asl_image = cv2.resize(asl_image, (asl_width, asl_height))

#-------------Webcam Setup------------------------------------------------
cap = cv2.VideoCapture(0)

# Customizable dimensions for webcam footage
webcam_width = 640
webcam_height = 500
cap.set(3, webcam_width)
cap.set(4, webcam_height)

#-------------Mediapipe Hands Setup---------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

#------------Setup Labels for Signs----------------------------------------
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
               9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
               17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

max_length = 42  # Maximum length as in the training data

#------------Green Rectangle-----------------------------------------------
rectangle_width = 115
rectangle_height = 100
rectangle_opacity = 0.5

#-----------Display Text----------------------------------------------------
display_text = ""
display_text_active = False
words_list = ["DAY", "HELLO", "WORLD", "FOOD", "WEEK", "MOOD"]
current_word_index = 0  # Index to track the current word
predicted_status = {word: False for word in words_list}

def update_display_text():
    global display_text, current_word_index, predicted_status
    display_text = words_list[current_word_index]
    predicted_status = {char: False for char in display_text.replace(" ", "")}

# Function to draw the display text
def draw_display_text(frame, text, status_dict):
    x = 50  # Starting position of the text (x-coordinate)
    y = 50  # y-coordinate
    for char in text:
        if char != " ":
            color = (0, 255, 0) if status_dict[char] else (0, 0, 255)  # Green if predicted, else red
            cv2.putText(frame, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            x += 40  # Move to the next position
        else:
            x += 20  # Space between words

#-----------Background Image----------------------------------------------------
background_image_path = 'asl-sign-language-detector-python\dojo_2.png'
background_image = cv2.imread(background_image_path)

# Resize the background image to be larger
bg_width, bg_height = 1500, 720  # Example dimensions, adjust as needed
background_image = cv2.resize(background_image, (bg_width, bg_height))

# Grid settings
grid_cols = 5
grid_rows = 5
move_x = asl_width // grid_cols
move_y = asl_height // grid_rows

predicted_character = None
asl_image_with_rectangle = asl_image.copy()

#-----------Background Music----------------------------------------------------
# Initialize pygame mixer
pygame.mixer.init()

# Load your music file
music_file_path = '[MapleStory BGM] Mu Lung Dojo 1.mp3'
pygame.mixer.music.load(music_file_path)

# Set the volume (0.0 to 1.0)
pygame.mixer.music.set_volume(0.25)

# Start playing the music in a loop
pygame.mixer.music.play(-1)  # -1 makes it play indefinitely

#-----------Display Score----------------------------------------------------
score = 0

def update_score():
    global score
    score += 1

# Function to check if the last letter of "D A Y" has changed and update the score
def check_and_update_score():
    global display_text, score, sequence_completed, predicted_status
    if all(predicted_status.values()) and not sequence_completed:
        update_score()
        sequence_completed = True

# Function to draw the score
def draw_score(frame, score):
    x = frame.shape[1] - 150  # Starting position of the score (x-coordinate)
    y = frame.shape[0] - 50   # y-coordinate
    cv2.putText(frame, f"Score: {score}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

# Function to draw the instruction text
def draw_instruction_text(frame, text):
    x = frame.shape[1] - 220  # Starting position of the text (x-coordinate)
    y = frame.shape[0] - 80   # y-coordinate
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

# Flag to track if the current sequence is completed
sequence_completed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (webcam_width, webcam_height))

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Initialize the variables for bounding box to None
    x1, y1, x2, y2 = None, None, None, None

    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]

            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 252, 3), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (44, 252, 3), 3,
                    cv2.LINE_AA)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.extend([x, y])

        data_aux.extend([0] * (max_length - len(data_aux)))

        if len(data_aux) == max_length:
            prediction = model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character in labels_dict.values():
                label_index = list(labels_dict.values()).index(predicted_character)

                col = label_index % grid_cols
                row = label_index // grid_cols

                # Compensate for the offset in the last two rows
                if row >= 3:
                    col = (col - 1) if col > 0 else grid_cols - 1

                rectangle_x = col * move_x
                rectangle_y = row * move_y

                asl_image_with_rectangle = asl_image.copy()
                overlay = asl_image_with_rectangle.copy()
                cv2.rectangle(overlay, (rectangle_x, rectangle_y),
                              (rectangle_x + rectangle_width, rectangle_y + rectangle_height),
                              (0, 255, 0), thickness=cv2.FILLED)
                cv2.addWeighted(overlay, rectangle_opacity, asl_image_with_rectangle, 1 - rectangle_opacity, 0, asl_image_with_rectangle)
    
    if predicted_character:
         # Update the prediction status
        if predicted_character in predicted_status:
            predicted_status[predicted_character] = True
        
    # Check for keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        print("Key 1 Pressed")
        sequence_completed = False
        update_display_text()
        display_text_active = True
        current_word_index = (current_word_index + 1) % len(words_list)
        
    elif key == ord('q'):
        break

     # Draw the display text if active
    if display_text_active:
        draw_display_text(frame, display_text, predicted_status)

    # Update the score if the last letter of "D A Y" has changed
    check_and_update_score()

    # Draw the instruction text on every frame
    draw_instruction_text(frame, "Press 1 for Words")

    # Draw the score on every frame
    draw_score(frame, score)

     # Center the webcam and ASL image on the background
    bg_center_x = bg_width // 2

    webcam_start_x = bg_center_x - (webcam_width + asl_width) // 2
    asl_start_x = webcam_start_x + webcam_width

    combined_frame_width = webcam_width + asl_width
    combined_frame_height = max(webcam_height, asl_height)
    combined_frame = np.zeros((combined_frame_height, combined_frame_width, 3), dtype=np.uint8)

    background_image[:webcam_height, webcam_start_x:webcam_start_x + webcam_width] = frame
    background_image[:asl_height, asl_start_x:asl_start_x + asl_width] = asl_image_with_rectangle


    cv2.imshow('frame', background_image)

cap.release()
cv2.destroyAllWindows()

# THIS COMMENTED OUT CODE IS THE BASE CODE FOR REPRODUCTION
# DANIAL C.

# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # --------------ASL Image-------------------------------------------------
# asl_image = cv2.imread('sign-language-detector-python/asl.png')
# desired_height = 300
# desired_width = 400
# resized_image = cv2.resize(asl_image, (desired_width, desired_height))
# alpha = 0.5

# #-------------Begin Webcam------------------------------------------------
# cap = cv2.VideoCapture(0)
# cap.set(3, 1250)  # Set width
# cap.set(4, 1250)  # Set height

# #-------------Setup Mediapipe Hands---------------------------------------
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# #------------Setup Labels for Signs----------------------------------------
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', \
#                9: 'J', 10: 'K', 11: 'L', 12:'M', 13: 'N', 14:'O', 15:'P', 16: 'Q', \
#                17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# max_length = 42  # Maximum length as in the training data

# while True:

#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     data_aux = []

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
            
#             x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
#             y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]

#             for landmark in hand_landmarks.landmark:
#                 x = landmark.x
#                 y = landmark.y
#                 data_aux.extend([x, y])

#         # Pad data_aux if it's shorter than max_length
#         data_aux.extend([0] * (max_length - len(data_aux)))

#         # Make prediction
#         if len(data_aux) == max_length:  # Ensure the data is of the correct length
#             prediction = model.predict([data_aux])
#             predicted_character = labels_dict[int(prediction[0])]

#             x1, y1 = int(min(x_coords)) - 10, int(min(y_coords)) - 10
#             x2, y2 = int(max(x_coords)) + 10, int(max(y_coords)) + 10

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 252, 3), 4)
#             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (44, 252, 3), 3,
#                         cv2.LINE_AA)
            
#     # Extract the region of interest (ROI) from the frame
#     roi = frame[10:10 + desired_height, 10:10 + desired_width]

#     # Blend the ROI with the resized image
#     blended = cv2.addWeighted(roi, 1 - alpha, resized_image, alpha, 0)

#     # Place the blended image back into the frame
#     frame[10:10 + desired_height, 10:10 + desired_width] = blended

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()