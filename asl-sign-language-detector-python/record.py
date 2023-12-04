import cv2
import time
import datetime

# Set the key you want to use to start recording (e.g., 'r' for 'record').
recording_keys = [ord('j'), ord('z'), ord('o')]

# Video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera (you can change it to the camera index you want)

# Variables for recording control
recording = False
frames = []

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while True:
    ret, frame = cap.read()
    flipped = cv2.flip(frame, 1)

    h, w, c = frame.shape
    pad = 10
    flipped = flipped[pad:h - pad, pad:w - pad]

    # Display the webcam input
    cv2.imshow('Webcam', flipped)

    key = cv2.waitKey(1) & 0xFF

    if key in recording_keys and not recording:
        print(f"Recording started for class {chr(key)}...")
        recording = True
        frames = []
        out = cv2.VideoWriter(f'clips/train/{chr(key)}/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.mp4', 
                              fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        start_time = time.time()

    if recording:
        frames.append(frame)
        if time.time() - start_time >= 2:
            recording = False
            if out:
                for f in frames:
                    out.write(f)
                out.release()
                print("Recording saved")
                out = None
            else:
                print("No frames captured during recording.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()