import os
import requests
from requests.auth import HTTPBasicAuth
from PIL import Image
import numpy as np
from config import url_template, username, password

output_folder = "D:\small_scroll_one"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def download_image(url, filename):
    print(url)
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    print(response.status_code)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download image {filename}, status code: {response.status_code}")
        return False

def resize_image(input_filename, output_filename, scale_factor):
    with Image.open(input_filename) as img:
        img = img.convert("I")  # Ensure image is in 32-bit mode
        img_array = np.array(img)
        
        # Scale pixel values from 16-bit to 8-bit depth
        img_array = (img_array / 256).astype(np.uint8)
        
        # Convert back to Image object and convert to greyscale
        img = Image.fromarray(img_array).convert("L")
        new_size = (img.width // scale_factor, img.height // scale_factor)
        resized_img = img.resize(new_size, Image.ANTIALIAS)
        resized_img.save(output_filename, format="JPEG")

for i in range(0, 7250, 8):
    original_filename = os.path.join(output_folder, f"0{i:04}.tif")
    resized_filename = os.path.join(output_folder, f"{i:04}_resized.jpg")
    url = url_template.format(i)
    
    if download_image(url, original_filename):
        resize_image(original_filename, resized_filename, 8)
        os.remove(original_filename)
