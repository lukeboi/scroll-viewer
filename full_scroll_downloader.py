import os
import requests
from requests.auth import HTTPBasicAuth
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from config import url_template, username, password

output_folder = "D:\small_scroll_one_from_server"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def download_image(url, filename):
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

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

def main(start, end, step):
    for i in tqdm(range(start, end, step)):
        original_filename = os.path.join(output_folder, f"0{i:04}.tif")
        resized_filename = os.path.join(output_folder, f"{i:04}_resized.jpg")
        url = url_template.format(i)

        # Check if output image already exists, and if so, skip download and resize
        if os.path.exists(resized_filename):
            print(f"Output image {resized_filename} already exists. Skipping download and resize.")
            continue

        if download_image(url, original_filename):
            pass
            # resize_image(original_filename, resized_filename, 8)
            # os.remove(original_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and resize images.')
    parser.add_argument('start', type=int, help='Start index')
    parser.add_argument('end', type=int, help='End index (exclusive)')
    parser.add_argument('step', type=int, help='Step size')
    args = parser.parse_args()

    main(args.start, args.end, args.step)