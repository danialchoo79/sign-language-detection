import numpy as np
import pandas as pd
import cv2
import os

def draw_image_from_coordinates(x, y, output_filename, width, height, line_thickness):
    # Create a blank white image with the specified dimensions
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # image.fill(255)  # Set the image to white

    # Convert coordinates to integer values
    x = np.array(x).astype(np.int32)
    y = np.array(y).astype(np.int32)

    # Draw the line on the image
    for i in range(len(x) - 1):
        cv2.line(image, (x[i], y[i]), (x[i+1], y[i+1]), (255, 255, 255), line_thickness)

    # Save the image
    cv2.imwrite(output_filename, image)

# Example coordinates
output_filename = "output_2_normalized.png"
width = 28
height = 28
line_thickness = 2

def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val) * 0.8 + 0.1
    return normalized_column

def smooth_column(column):
    smoothed_column = column.rolling(window=3, min_periods=1).mean()
    return smoothed_column

files = os.listdir('train/z/')
df = pd.read_csv(f'train/z/{files[5]}')
df = df.apply(normalize_column, axis=0)
df = df.apply(smooth_column, axis=0)

x_coordinates = (df['center_x'] * width).astype('int32')
y_coordinates = (df['center_y'] * height).astype('int32')

print(x_coordinates)
print(y_coordinates)

draw_image_from_coordinates(x_coordinates, y_coordinates, output_filename, width, height, line_thickness)