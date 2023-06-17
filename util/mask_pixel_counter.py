import os
import cv2
import numpy as np

root_dir = 'S:\\server_uploads\\segmentations'  # Please replace with your directory
total_pixel_value_255 = 0

def count_255_pixels_in_image(image_path):
    # Load image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Count pixels with 255 value
    pixel_value_255_count = np.sum(img == 255)
    
    return pixel_value_255_count

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "mask.png":
            file_path = os.path.join(subdir, file)
            total_pixel_value_255 += count_255_pixels_in_image(file_path)

print(f'Total pixels with 255 value: {total_pixel_value_255}')
print(f'Which suggests a total cm^2 area of: {total_pixel_value_255 * 8 * 8 * 1e-8}')