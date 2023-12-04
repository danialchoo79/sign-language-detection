import numpy as np
import pandas as pd
import cv2
import os


def draw_image_from_coordinates(x, y, width, height, line_thickness):
    # Create a blank white image with the specified dimensions
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert coordinates to integer values
    x = np.array(x).astype(np.int32) # flip horizontally
    y = np.array(y).astype(np.int32)

    # Draw the line on the image
    for i in range(len(x) - 1):
        cv2.line(image, (x[i], y[i]), (x[i+1], y[i+1]), (255, 255, 255), line_thickness)

    # Save the image
    return image

def smooth_column(column):
    smoothed_column = column.rolling(window=3, min_periods=1).mean()
    return smoothed_column

def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    norm_range = max(0.25, max_val - min_val)
    normalized_column = (column - min_val) / (norm_range) * 0.8 + 0.1
    return normalized_column


# Example coordinates
smooth_window = 5
width = 28
height = 28
line_thickness = 2

in_root = 'sequences/train'
out_root = 'images/train'
for folder in ['j', 'o', 'z']:
    print('Processing folder: ' + folder)
    for filename in os.listdir(f'{in_root}/{folder}'):
        df = pd.read_csv(f'{in_root}/{folder}/{filename}')
        df = df.apply(lambda row: row.fillna(df.ffill().mean() + df.bfill().mean()) / 2, axis=1)
        df = df.apply(smooth_column, axis=0)
        df = df.apply(normalize_column, axis=0)

        x_coordinates = (df['center_x'] * width).astype('int32')
        y_coordinates = (df['center_y'] * height).astype('int32')

        img = draw_image_from_coordinates(x_coordinates, y_coordinates, width, height, line_thickness)
        cv2.imwrite(f'{out_root}/{folder}/{filename[:-4]}.png', img)

