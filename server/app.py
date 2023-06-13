import time
import numpy as np
from io import BytesIO
from flask import Flask, request, send_file, make_response
from flask_cors import CORS, cross_origin
import json
import traceback
import os
from PIL import Image, ImageOps
import tifffile as tiff
import struct
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import tqdm
import concurrent.futures
import requests
from config import username, password
from requests.auth import HTTPBasicAuth
import traceback
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage import measure
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
import imageio
from skspatial.objects import Plane, Points
from scipy.interpolate import griddata
from scipy import isnan
from scipy.interpolate import RegularGridInterpolator

from converttoraw import convert_tif_stack_to_raw

app = Flask(__name__)
CORS(app, supports_credentials=True)

json_file = os.path.abspath("server/config.json")

volume = None
volume_size = None

# global status message for the user
server_status = "Server started"

# Function to create a 3D Gaussian kernel
def get_gaussian_kernel(size=3, sigma=2.0, channels=1):
    # Create a vector of size 'size' filled with 'size' evenly spaced values from -size//2 to size//2
    x_coord = torch.arange(start=-size//2, end=size//2 + 1, dtype=torch.float)
    # Create a 3D grid of size 'size' x 'size' x 'size'
    x, y, z = torch.meshgrid(x_coord, x_coord, x_coord)
    # Calculate the 3D Gaussian kernel
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2*sigma**2))
    # Normalize the kernel
    kernel = kernel / torch.sum(kernel)
    return kernel.float()

# Function to create a 3D convolution layer with a Gaussian kernel
def gaussian_blur3d(channels=1, size=3, sigma=2.0):
    kernel = get_gaussian_kernel(size, sigma, channels)
    # Repeat the kernel for all input channels
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    # Create a convolution layer
    blur_layer = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=size, groups=channels, bias=False, padding='same')
    # Set the kernel weights
    blur_layer.weight.data = nn.Parameter(kernel)
    # Make the layer non-trainable
    blur_layer.weight.requires_grad = False
    return blur_layer

def sobel_filter_3d(input, chunks=4, overlap=3, return_vectors=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define 3x3x3 kernels for Sobel operator in 3D
    sobel_x = torch.tensor([
        [[[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]],
         [[ 2, 0, -2], [ 4, 0, -4], [ 2, 0, -2]],
         [[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]]],
    ], dtype=torch.float32).to(device)

    sobel_y = sobel_x.transpose(2, 3)
    sobel_z = sobel_x.transpose(1, 3)

    # Add an extra dimension for the input channels
    sobel_x = sobel_x[None, ...]
    sobel_y = sobel_y[None, ...]
    sobel_z = sobel_z[None, ...]

    assert len(input.shape) == 5, "Expected 5D input (batch_size, channels, depth, height, width)"

    depth = input.shape[2]
    chunk_size = depth // chunks
    chunk_overlap = overlap // 2

    results = []
    vectors = []

    for i in range(chunks):
        # Determine the start and end index of the chunk
        start = max(0, i * chunk_size - chunk_overlap)
        end = min(depth, (i + 1) * chunk_size + chunk_overlap)

        if i == chunks - 1:  # Adjust the end index for the last chunk
            end = depth

        chunk = input[:, :, start:end, :, :]

        # Move chunk to GPU
        chunk = chunk.to(device)

        G_x = nn.functional.conv3d(chunk, sobel_x, padding=1)
        G_y = nn.functional.conv3d(chunk, sobel_y, padding=1)
        G_z = nn.functional.conv3d(chunk, sobel_z, padding=1)

        # Compute the gradient magnitude
        G = torch.sqrt(G_x ** 2 + G_y ** 2 + G_z ** 2)

        # Remove the overlap from the results
        if i != 0:  # Not the first chunk
            G = G[:, :, chunk_overlap:, :, :]
            G_x = G_x[:, :, chunk_overlap:, :, :]
            G_y = G_y[:, :, chunk_overlap:, :, :]
            G_z = G_z[:, :, chunk_overlap:, :, :]
        if i != chunks - 1:  # Not the last chunk
            G = G[:, :, :-chunk_overlap, :, :]
            G_x = G_x[:, :, :-chunk_overlap, :, :]
            G_y = G_y[:, :, :-chunk_overlap, :, :]
            G_z = G_z[:, :, :-chunk_overlap, :, :]

        if return_vectors:
            vector = torch.stack((G_x, G_y, G_z), dim=5)
            vectors.append(vector.cpu())
            
        # Move the result back to CPU and add it to the list
        results.append(G.cpu())

        # Free memory of intermediate variables
        del G_x, G_y, G_z, chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate the results along the depth dimension
    result = torch.cat(results, dim=2)

    if vectors:
        vector = torch.cat(vectors, dim=2)

    if return_vectors:
        return result, vector
    return result

# Downloads a single image into a folder.
def download_image_in_folder(i, output_folder, url_template, extension="tif"):
    original_filename = os.path.join(output_folder, f"0{i:04}.{extension}")
    url = url_template.format(i)

    # Check if output image already exists, and if so, skip download
    if os.path.exists(original_filename):
        print(f"Output image {original_filename} already exists. Skipping download.")
        return

    try:
        response = requests.get(url, auth=HTTPBasicAuth(username, password))

        if response.status_code == 200:
            with open(original_filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {original_filename}")
        else:
            return f"Failed to download image {original_filename}, status code: {response.status_code}"
    except Exception as e:
        return f"Error downloading image {original_filename}: {str(e)}"

    return None  # Return None if no error occurred


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
        
def resize_images_into_another_folder(src, dest, scale, depth="16"):
    # Check if the data is already downloaded
    files_exist = False
    folder_exists = os.path.exists(dest)

    # If the folder exists, see if it has the correct number of files
    if folder_exists:
        files_exist = len(os.listdir(dest)) == len(os.listdir(src))

    if files_exist:
        return
    
    for src_img in os.listdir(src):
        # Get the filename without the extension
        filename, ext = os.path.splitext(src_img)
        # Create the output folder path by joining the destination directory and the filename
        output_folder = os.path.join(dest, filename)
        # Create the output file path by joining the output folder and the new extension
        output_file = output_folder + ".jpg"
        # Skip if the output file already exists
        if os.path.exists(output_file):
            continue
        # Resize the image and save it with the new extension
        resize_image(os.path.join(src, src_img), output_file, scale) # depth


# Download a single folder in the data_srcs
def download_single_imageset_into_folder(metadata):
    assert metadata["url"]
    assert metadata["folder"]
    assert metadata["num_to_download"]

    print("Checking if data exists in", metadata["folder"], end=" ... ")

    # Check if the data is already downloaded
    files_exist = False
    folder_exists = os.path.exists(metadata["folder"])

    # If the folder exists, see if it has the correct number of files
    if folder_exists:
        files_exist = len(os.listdir(metadata["folder"])) == metadata["num_to_download"]

    # If the data isn't downloaded, downlaod it
    if files_exist:
        print("Exists!")

    else:
        print("Does not exist. Downloading...")
        if not folder_exists:
            os.makedirs(metadata["folder"])

        start = 0
        end = metadata["num_to_download"] - 1

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda i: download_image_in_folder(i, metadata["folder"], metadata["url"]), range(start, end + 1)), total=metadata["num_to_download"]))

        # Check for errors in the results
        errors = [result for result in results if result is not None]
        if errors:
            print(f"Errors occurred while downloading data: {errors}")

def verify_config_is_downloaded(config_filename):
    # Verify all full size slices are downloaded
    download_single_imageset_into_folder({
        "url": config_filename["src"],
        "folder": config_filename["full_dl"],
        "num_to_download": config_filename["dimensions"][2]
    })
    os.makedirs(config_filename["full_dl"], exist_ok=True)
    os.makedirs(config_filename["two_jpg"], exist_ok=True)
    os.makedirs(config_filename["sixteen_jpg"], exist_ok=True)
    tif_files = [f for f in os.listdir(config_filename["full_dl"])]

    two_files = [f for f in os.listdir(config_filename["two_jpg"])]
    if len(two_files) < config_filename["dimensions"][2]:
        # create the half size files
        resize_images_into_another_folder(config_filename["full_dl"], config_filename["two_jpg"], 2)


    sixteen_files = [f for f in os.listdir(config_filename["sixteen_jpg"]) if f.lower()]
    if len(sixteen_files) < config_filename["dimensions"][2]:
        # create the half size files
        resize_images_into_another_folder(config_filename["full_dl"], config_filename["two_jpg"], 16)


def get_volume_from_tif_stack(src, origin, size, lod_downsample, depth="8", extension="tif", threshold=0):
    # Get the list of TIF files in the input directory
    tif_files = [f for f in os.listdir(src) if f.lower().endswith(f".{extension}")]

    print(f".{extension}")

    # https://chat.openai.com/c/08da5b37-f7ed-4130-a511-da77aeff91d6
    # tif_files.sort(key=lambda x: int(x.split('.')[0]))
    tif_files.sort(key=lambda x: int(re.sub(r'\D', '', x.split('.')[0])))

    global server_status
    server_status = "Loading: "

    # Create an empty bytearray to store the .raw data
    # raw_data = bytearray(width * height * len(tif_files))
    shape = (size[0] // lod_downsample, size[1] // lod_downsample, (size[2] // lod_downsample))
    data = np.zeros(shape, np.uint8)
    data_unthresholded = np.zeros(shape, np.uint8)

    # Iterate over the TIF files, converting them to R8 format and adding them to the raw_data bytearray
    # Skip 
    print(tif_files)
    for i, tif_file in enumerate(tif_files[origin[2]:origin[2] + size[2]:lod_downsample]):
        image_path = os.path.join(src, tif_file)
        # image = Image.open(image_path)
        image = None
        if "tif" in extension:
            image = tiff.imread(image_path)
        else:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            image = np.array(image)
            image = np.transpose(image, (1, 0))

        # print(np.array(image).shape)
        # Convert the image to grayscale (single channel, 8 bits)
        # gray_image = image.convert("L")
        # gray_image.show()

        # print(image.shape)
        # print(image.dtype)

        # Crop the image
        # gray_image = gray_image.crop((origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]))
        image = image[origin[0]:origin[0] + size[0], origin[1]:origin[1] + size[1]]
        
        # Downsample the image
        image = image[::lod_downsample, ::lod_downsample]

        # print(np.array(image).mean())
        
        if depth == "16":
            image = (image * (255 / 65535)).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)
        # image = image.astype(np.uint8)

        # print(np.array(image).shape)

        data_unthresholded[:, :, i] = image

        # Apply threshold
        # threshold = 0
        image[image < threshold] = 0
        image[image >= threshold] = ((image[image >= threshold] - threshold) / (255 - threshold)) * 255

        print(i)

        data[:, :, i] = image

        server_status = f"Loading: {i}/{len(tif_files[origin[2]:origin[2] + size[2]])}"

    server_status = "Done Loading!"

    # raw_data = raw_data.transpose(raw_data, (2, 1, 0))

    return data, data_unthresholded, shape

# funny little heartbeat
heartbeat_counter = 0
heartbeat_threshold = 10

@app.route('/heartbeat', methods=['GET'])
# @cross_origin(supports_credentials=True)
def get_heartbeat():
    global heartbeat_counter
    heart = "< 3" if heartbeat_counter % heartbeat_threshold == 0 else "<3"
    heartbeat_counter += 1

    response = make_response(heart + "<br>" + server_status, 200)
    response.mimetype = "text/plain"
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

# @cross_origin(supports_credentials=True)
@app.route('/volume_metadata', methods=['GET'])
def get_volume_metadata():
    global json_file
    return send_file(json_file)

# @cross_origin(supports_credentials=True)
@app.route('/volume', methods=['GET'])
def volume():
    try:
        filename = request.args.get('filename') # Requested filename for lookup in config.json

        # Size and origin of request, in original resoultion.
        size = [int(x) for x in request.args.get('size').split(',')]
        origin = [int(x) for x in request.args.get('origin').split(',')]
        cutoffPlane = [float(x) for x in request.args.get('cutoffPlane').split(',')]

        apply_sobel = request.args.get('applySobel') == "true"

        lod_downsample = 1

        # Load the config
        config = None
        with open(json_file, 'r') as f:
            config = json.load(f)

        # Volume information
        global volume
        global volume_size

        volume = None # 3d data
        volume_size = None # 3d size, not always the same size as above due to LOD scaling.

        if filename == "random_volume":
            # Generate the random 3D volume with values in the range 0 to 255 (8-bit)
            volume = np.random.randint(0, 256, size=(size[0], size[1], size[2]), dtype=np.uint8)
            volume = volume.tobytes()

            volume_size = size
            
        elif filename == "line":
            volume = np.zeros((size), dtype=np.uint8)

            # Calculate the steps needed to reach the opposite corner
            steps = np.array([size[0] - 1, size[1] - 1, size[2] - 1], dtype=np.float32)

            # Calculate the number of points in the line
            num_points = int(np.ceil(np.max(steps)))

            # Calculate the step sizes along each axis
            step_sizes = steps / num_points

            # Set the points along the line to 1
            for i in range(num_points + 1):
                point = (i * step_sizes).astype(int)
                volume[point[0], point[1], point[2]] = 255

            volume = np.transpose(volume, (2, 1, 0))
            volume = volume.tobytes()

            volume_size = size
            # print(volume)
            # volume = np.asfortranarray(volume)

        elif filename == "ball":
            print("BALL BALL BALL")
            
            volume_size = size

            # Dimensions of the 3D array
            depth, height, width = size

            # Create an empty 3D NumPy array with uint8 dtype
            volume = np.zeros((depth, height, width), dtype=np.uint8)

            # Generate 3D grid coordinates
            z_coords, y_coords, x_coords = np.ogrid[:depth, :height, :width]

            # Center coordinates of the ellipse
            center_x = width // 2
            center_y = height // 2
            center_z = depth // 2

            # Radii of the ellipse
            radius_x = width // 3
            radius_y = height // 3
            radius_z = depth // 3

            # Calculate the squared distances from the center
            distances = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2 + ((z_coords - center_z) / radius_z) ** 2

            # Use a threshold to determine indices inside the ellipse
            indices = distances <= 1

            # Set the values to 255 (white) for indices inside the ellipse
            volume[indices] = 255

            # Create a Gaussian blur layer
            blur_layer = gaussian_blur3d(channels=1, size=3, sigma=2.0)

            # Apply the Gaussian blur to the image
            # volume = blur_layer(torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)).numpy().astype(np.uint8).squeeze(0).squeeze(0)

            volume = sobel_filter_3d(torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)).numpy().astype(np.uint8).squeeze(0).squeeze(0)

            # cpu gaussian fliter. lol.
            volume = gaussian_filter(volume * 255, sigma=1).astype(np.uint8) > 0
            sobel_output = sobel_filter_3d(torch.from_numpy((volume).astype(np.float32)).unsqueeze(0).unsqueeze(0), return_vectors=True)
            volume = sobel_output[0].numpy().astype(np.uint8).squeeze(0).squeeze(0)
            sobel_vectors = sobel_output[1].squeeze(0).squeeze(0)

            # set the top and bottom layers of the sobel vectors to zero.
            sobel_vectors[:, :, 0] = [0, 0, 0]
            sobel_vectors[:, :, -1] = [0, 0, 0]

            factor = 10

            # Create an array to hold the average vectors
            averages = np.zeros((size[0]//factor, size[1]//factor, size[2]//factor, 3))

            # Iterate over each 10x10x10 chunk
            for i in range(volume_size[0]//factor):
                print(i)
                for j in range(volume_size[1]//factor):
                    for k in range(volume_size[2]//factor):
                        chunk = sobel_vectors[factor*i:factor*(i+1), factor*j:factor*(j+1), factor*k:factor*(k+1)]
                        norms = np.linalg.norm(chunk, axis=-1, keepdims=True)
                        normalized_vecs = np.divide(chunk, norms, where=norms!=0)
                        vec_sum = np.nansum(normalized_vecs, axis=(0,1,2))
                        count = np.count_nonzero(norms)
                        vec_avg = vec_sum / count if count > 0 else np.zeros(3)
                        vec_avg = vec_avg / np.linalg.norm(vec_avg) if np.linalg.norm(vec_avg) > 0 else np.zeros(3)
                        averages[i, j, k] = vec_avg

            # Split the 'averages' array into three 3D arrays
            x_avg = averages[:, :, :, 0]
            y_avg = averages[:, :, :, 1]
            z_avg = averages[:, :, :, 2]

            volume = (x_avg * 254).astype(np.uint8)
            volume_size = x_avg.shape

            volume = np.transpose(volume, (2, 1, 0))
            volume = volume.tobytes()
            print("done")

        else:
            print(config)
            # Load that volume
            if filename in config:
                # load from a tif stack
                print("downloading")

                if config[filename]["datasrc"] == "url":
                    verify_config_is_downloaded(config[filename])

                print("loading")
                volume_p, volume_no_threshold, volume_size = get_volume_from_tif_stack(config[filename]["full_dl"], origin, size,
                                                     lod_downsample=1,
                                                     depth=config[filename]["depth"],
                                                     extension=config[filename]["extension"],
                                                     threshold=int(request.args.get('threshold')))
                
                if apply_sobel:
                    volume_p = sobel_filter_3d(torch.from_numpy(volume_p.astype(np.float32)).unsqueeze(0).unsqueeze(0)).numpy().astype(np.uint8).squeeze(0).squeeze(0) > 1
                    # volume_p = sobel_filter_3d(torch.from_numpy(volume_p.astype(np.float32)).unsqueeze(0).unsqueeze(0)).numpy().astype(np.uint8).squeeze(0).squeeze(0)
                    volume_p = (volume_p * (254 / volume_p.max())).astype(np.uint8)

                    # Erode and dilute once, to get rid of most noise bubbles
                    kernel = np.ones((3, 3, 10), dtype=np.uint8)
                    volume_p = binary_erosion(volume_p, structure=kernel)
                    volume_p = binary_dilation(volume_p, structure=kernel)

                    # Dilute again, to make the actual segments really big and whole
                    kernel = np.ones((4, 4, 4), dtype=np.uint8)
                    volume_p = binary_dilation(volume_p, structure=kernel)

                    # Erode and dilute again, gets rid of a lot of the last noise
                    kernel = np.ones((6, 6, 15), dtype=np.uint8)
                    volume_p = binary_erosion(volume_p, structure=kernel)
                    volume_p = binary_dilation(volume_p, structure=kernel)

                    # Isolate just the segment at the centerpoint
                    labels3d = measure.label(volume_p)
                    start_pixel = [volume_p.shape[0] // 2, volume_p.shape[1] // 2, volume_p.shape[2] // 2]
                    target_value = labels3d[start_pixel[0], start_pixel[1], start_pixel[2]]
                    isolated_volume = np.where(labels3d == target_value, volume_no_threshold, 0)
                    volume_p = np.where(labels3d == target_value, volume_p, 0)

                    # cpu gaussian fliter. lol.
                    volume_p = gaussian_filter(volume_p * 255, sigma=1).astype(np.uint8) > 0
                    sobel_output = sobel_filter_3d(torch.from_numpy((volume_p).astype(np.float32)).unsqueeze(0).unsqueeze(0), return_vectors=True)
                    volume_p = sobel_output[0].numpy().astype(np.uint8).squeeze(0).squeeze(0)
                    sobel_vectors = sobel_output[1].squeeze(0).squeeze(0)

                    if False:
                        volume_p = isolated_volume

                    if False:
                        volume_p = (volume_p * 128).astype(np.uint8) + (isolated_volume / 2).astype(np.uint8)
                    
                    # # volume_p = # SCale down by 10x
                    # # Calculate the dot product of each vector with the plane normal
                    # dotProducts = np.tensordot(sobel_vectors, [-1, -1, 0], axes=([-1], [0]))
                    
                    # # Identify indices where the dot product is negative (vector is facing opposite to the plane)
                    # opposite_indices = np.where(dotProducts < 0)

                    # # Set these vectors to zero
                    # volume_p[opposite_indices] = 0

                    # Average the vectors
                    # Determine the size of the larger array

                    # set the top and bottom layers of the sobel vectors to zero.
                    sobel_vectors[:, :, 0] = torch.tensor([0, 0, 0])
                    sobel_vectors[:, :, -1] = torch.tensor([0, 0, 0])
                    volume_p[:, :, 0] = torch.tensor(0)
                    volume_p[:, :, -1] = torch.tensor(0)
                    volume_p[:, 0, :] = torch.tensor(0)
                    volume_p[:, -1, :] = torch.tensor(0)
                    volume_p[0, :, :] = torch.tensor(0)
                    volume_p[-1, :, :] = torch.tensor(0)
                    
                    # Take an average downsampling
                    # factor = 10

                    # # Create an array to hold the average vectors
                    # averages = np.zeros((size[0]//factor, size[1]//factor, size[2]//factor, 3))

                    # # Iterate over each 10x10x10 chunk
                    # for i in range(volume_size[0]//factor):
                    #     print(i)
                    #     for j in range(volume_size[1]//factor):
                    #         for k in range(volume_size[2]//factor):
                    #             chunk = sobel_vectors[factor*i:factor*(i+1), factor*j:factor*(j+1), factor*k:factor*(k+1)]
                    #             norms = np.linalg.norm(chunk, axis=-1, keepdims=True)
                    #             normalized_vecs = np.divide(chunk, norms, where=norms!=0)
                    #             vec_sum = np.nansum(normalized_vecs, axis=(0,1,2))
                    #             count = np.count_nonzero(norms)
                    #             vec_avg = vec_sum / count if count > 0 else np.zeros(3)
                    #             vec_avg = vec_avg / np.linalg.norm(vec_avg) if np.linalg.norm(vec_avg) > 0 else np.zeros(3)
                    #             averages[i, j, k] = vec_avg

                    # # print the gradients for easy copy pasting into the line stuff on the frontend
                    # for x in range(averages.shape[0]):
                    #     for y in range(averages.shape[1]):
                    #         for z in range(averages.shape[2]):
                    #             if np.any(averages[x][y][z]):  # check if nonzero
                    #                 position = [x / averages.shape[0], y / averages.shape[1], z / averages.shape[2]]
                    #                 print(*position, *(averages[x][y][z] / 10 + position), " ", sep=",")
                    # # # Split the 'averages' array into three 3D arrays
                    # x_avg = averages[:, :, :, 0]
                    # # y_avg = averages[:, :, :, 1]
                    # # z_avg = averages[:, :, :, 2]

                    # Throw out sobel vectors which aren't in the right direciton
                    # # Calculate the cosine of the maximum angle
                    max_angle_rad = np.radians(90)
                    cos_max_angle = np.cos(max_angle_rad)
                    
                    # Calculate the dot product of each vector with the reference vector
                    dotProducts = np.tensordot(sobel_vectors, cutoffPlane, axes=([-1], [0]))
                    
                    # Calculate the magnitudes of the vectors in vectorArray and the reference vector
                    magnitudes = np.linalg.norm(sobel_vectors, axis=-1) * np.linalg.norm(cutoffPlane)
                    
                    # Calculate the cosines of the angles
                    cos_angles = dotProducts / magnitudes
                    
                    # Identify indices where the cosine of the angle is less than the cosine of the maximum angle
                    # (i.e., the angle is larger than the maximum angle)
                    invalid_indices = np.where(cos_angles < cos_max_angle)

                    volume_p[invalid_indices] = torch.tensor([0])
                    
                    # Erode and dilute the array, making a mask of vectors to keep
                    kernel = np.ones((1, 1, 8), dtype=np.uint8)
                    mask_of_vectors_to_keep = binary_erosion(volume_p, structure=kernel)
                    mask_of_vectors_to_keep = binary_dilation(mask_of_vectors_to_keep, structure=kernel)
                    mask_of_vectors_to_keep = binary_dilation(mask_of_vectors_to_keep, structure=kernel)

                    
                    # Create the 3D coordinates
                    x, y, z = np.meshgrid(np.arange(mask_of_vectors_to_keep.shape[0]), np.arange(mask_of_vectors_to_keep.shape[1]), np.arange(mask_of_vectors_to_keep.shape[2]))

                    # Find the non-zero points
                    non_zero_points = np.where(mask_of_vectors_to_keep != 0)
                    
                    non_zero_points = (
                        non_zero_points[0][::200],
                        non_zero_points[1][::200],
                        non_zero_points[2][::200]
                    )

                    vector_positions = []
                    vector_normals = []
                    # Create a flag for the first plane
                    first_plane = True

                    chunk = 15

                    # Iterate over non-zero points
                    for nx, ny, nz in tqdm.tqdm(zip(*non_zero_points), total=len(non_zero_points[0])):
                        # Create a 10x10x10 subgrid around the point, ensure it's within array bounds
                        subgrid_x_start, subgrid_x_end = max(0, nx - chunk), min(mask_of_vectors_to_keep.shape[0], nx + chunk)
                        subgrid_y_start, subgrid_y_end = max(0, ny - chunk), min(mask_of_vectors_to_keep.shape[1], ny + chunk)
                        subgrid_z_start, subgrid_z_end = max(0, nz - chunk), min(mask_of_vectors_to_keep.shape[2], nz + chunk)

                        # Use these uniform ranges to slice x, y, z coordinates and the values
                        subgrid_x = x[subgrid_x_start:subgrid_x_end, subgrid_y_start:subgrid_y_end, subgrid_z_start:subgrid_z_end]
                        subgrid_y = y[subgrid_x_start:subgrid_x_end, subgrid_y_start:subgrid_y_end, subgrid_z_start:subgrid_z_end]
                        subgrid_z = z[subgrid_x_start:subgrid_x_end, subgrid_y_start:subgrid_y_end, subgrid_z_start:subgrid_z_end]
                        values = mask_of_vectors_to_keep[subgrid_x_start:subgrid_x_end, subgrid_y_start:subgrid_y_end, subgrid_z_start:subgrid_z_end]

                        # Filter out zeros
                        non_zero_subgrid = np.where(values != 0)

                        # If there are fewer than 3 points, we can't fit a plane
                        if non_zero_subgrid[0].size < 3:
                            continue

                        # Create skspatial Points object
                        points = Points(np.stack([subgrid_x[non_zero_subgrid], subgrid_y[non_zero_subgrid], subgrid_z[non_zero_subgrid]]).T)

                        points = points.mean_center()

                        # Fit plane to points
                        plane = Plane.best_fit(points)

                        volume_p[nx, ny, nz] = 255

                        normal = plane.normal if np.dot(plane.normal, cutoffPlane) > np.dot(-plane.normal, cutoffPlane) else -plane.normal

                        vector_positions.append([nx, ny, nz])
                        vector_normals.append(normal)

                        # Plane normal gives the direction of best fit
                        # print(plane.vector)

                        # If this is the first plane, create the plot
                        if first_plane:
                            fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
                            
                            points.plot_3d(ax, s=75, depthshade=False)
                            plane.plot_3d(ax, alpha=0.5, lims_x=(-10, 10), lims_y=(-10, 10))
                            ax.plot([0, cutoffPlane[0] * 50], [0, cutoffPlane[1] * 50], [0, cutoffPlane[2] * 50])
                            ax.plot([0, plane.normal[0] * 50], [0, plane.normal[1] * 50], [0, plane.normal[2] * 50])

                            # Set axes limits
                            # ax.set_xlim([np.min(x), np.max(x)])
                            # ax.set_ylim([np.min(y), np.max(y)])
                            # ax.set_zlim([np.min(z), np.max(z)])
                            ax.set_xlim([-30, 30])
                            ax.set_ylim([-30, 30])
                            ax.set_zlim([-30, 30])
                            
                            plt.savefig("plot.png")  # Save figure to an image file
                            first_plane = False

                    # plot the gradients
                    fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(18, 18))

                    points = Points(vector_positions)
                    points, center = points.mean_center(return_centroid=True)
                    points.plot_3d(ax, s=75, depthshade=False)
                    # plane.plot_3d(ax, alpha=0.5, lims_x=(-10, 10), lims_y=(-10, 10))
                    ax.plot([0, cutoffPlane[0] * 250], [0, cutoffPlane[1] * 250], [0, cutoffPlane[2] * 250])
                    # ax.plot([0, plane.normal[0] * 50], [0, plane.normal[1] * 50], [0, plane.normal[2] * 50])

                    for i in range(len(vector_positions)):
                        ax.plot(
                            [vector_positions[i][0] - center[0], vector_positions[i][0] - center[0] + vector_normals[i][0] * 25],
                            [vector_positions[i][1] - center[1], vector_positions[i][1] - center[1] + vector_normals[i][1] * 25],
                            [vector_positions[i][2] - center[2], vector_positions[i][2] - center[2] + vector_normals[i][2] * 25],
                        )

                    # Set axes limits
                    ax.set_xlim([-150, 150])
                    ax.set_ylim([-150, 150])
                    ax.set_zlim([-50, 50])
                    
                    plt.savefig("plot.png")  # Save figure to an image file

                    # Cool 3d rotating volume
                    # # Rotate the axes and update
                    # rot_animation = []
                    # for angle in range(0, 360, 10): # rotating by 10 degrees in each frame
                    #     ax.view_init(elev=40, azim=angle)

                    #     # save current figure as an image in the list
                    #     fname = 'tmp/rotate_plot_'+str(angle)+'.png'
                    #     plt.savefig(fname)
                    #     rot_animation.append(imageio.imread(fname))

                    # imageio.mimsave('rotating_plot.gif', rot_animation)
                    

                    plane = Plane.best_fit(points)
                    # Project points onto the plane
                    projected_points = [plane.project_point(point) for point in points]

                    # Convert list of Points to numpy array
                    projected_points = np.vstack([point for point in projected_points])

                    # Find the two vectors that span the plane (orthogonal to normal)
                    u = np.cross(plane.normal, np.array([1, 0, 0]))
                    if np.allclose(u, 0):  # if plane normal and [1, 0, 0] are parallel
                        u = np.cross(plane.normal, np.array([0, 1, 0]))
                    u = u / np.linalg.norm(u)  # normalize
                    v = np.cross(plane.normal, u)
                    v = v / np.linalg.norm(v)  # normalize

                    # Transform 3D coordinates to 2D coordinates in the plane
                    coordinates_2d = []
                    z_distances = []
                    for orig_point, proj_point in zip(points, projected_points):
                        coordinates_2d.append([np.dot(proj_point, u), np.dot(proj_point, v)])
                        z_distances.append(plane.distance_point_signed(orig_point))  # append the distance of the point to the plane

                    coordinates_2d = np.array(coordinates_2d)

                    # Find min/max 2D coordinates to get the dimensions of the bounding box
                    x_min, y_min = np.min(coordinates_2d, axis=0)
                    x_max, y_max = np.max(coordinates_2d, axis=0)

                    # The XY dimensions of the plane
                    plane_x = x_max - x_min
                    plane_y = y_max - y_min

                    print(f"Plane dimensions: X = {plane_x}, Y = {plane_y}")
                    print()


                    # Sample points is of shape position then direciton.
                    # X, Y, Z, x, y, z
                    # Shape is width, height, 6
                    unstretched_sample_points = np.zeros([int(np.ceil(plane_x)), int(np.ceil(plane_y)), 6])
                    for i in range(len(coordinates_2d)):
                        print(i)
                        unstretched_sample_points[int(abs(x_min) + coordinates_2d[i][0]), int(abs(y_min) + coordinates_2d[i][1])] = np.concatenate([points[i], vector_normals[i]])

                    print()

                    fig, ax = plt.subplots(figsize=(30, 30))
                    plt.imshow(unstretched_sample_points[:,:,0])
                    plt.colorbar()
                    plt.savefig("sample_points_plot.png")


                    # Interpolate the xyz coordinates
                    # Assuming 'coordinates' is your 2D array
                    rows, cols, _ = unstretched_sample_points.shape

                    # Create 2D indices arrays for your rows and columns
                    row_indices, col_indices = np.indices((rows, cols))

                    # Flatten all arrays for easier processing
                    flattened_coordinates = unstretched_sample_points.reshape(-1, 6)
                    flattened_row_indices = row_indices.flatten()
                    flattened_col_indices = col_indices.flatten()

                    # Create an array of the flattened 2D indices
                    flattened_2d_indices = np.vstack((flattened_row_indices, flattened_col_indices)).T

                    # Remove zero coordinates
                    non_zero_indices = np.any(flattened_coordinates != 0, axis=1)

                    non_zero_coordinates = flattened_coordinates[non_zero_indices]
                    non_zero_2d_indices = flattened_2d_indices[non_zero_indices]

                    # Generate a dense grid for interpolation
                    dense_row_indices = np.linspace(0, rows-1, rows)
                    dense_col_indices = np.linspace(0, cols-1, cols)

                    dense_grid = np.meshgrid(dense_row_indices, dense_col_indices)

                    # Perform cubic interpolation
                    interpolated_coordinates = griddata(non_zero_2d_indices, non_zero_coordinates, (dense_grid[0], dense_grid[1]), method='cubic')



                    fig, ax = plt.subplots(figsize=(30, 30))
                    plt.imshow(interpolated_coordinates[:,:,4].T)
                    plt.colorbar()
                    plt.savefig("sample_points_plot_inter_pre_norm.png")


                    # Now that the values have been interpolated, normalize the unormalized direction vectors which are in the last three values

                    # Split your array into two parts: the first three components and the last three components
                    first_three_components = interpolated_coordinates[:, :, :3]
                    last_three_components = interpolated_coordinates[:, :, 3:]

                    # Compute the norm of the last three components along the last dimension
                    norms = np.linalg.norm(last_three_components, axis=2)

                    # Identify indices where norms are zero or direction vectors are NaN
                    invalid_indices = np.logical_or(norms==0, np.isnan(norms))

                    # To avoid division by zero, set the norm to 1 where it's zero or NaN
                    norms[invalid_indices] = 1

                    # Normalize the last three components by dividing them with their respective norms
                    # We add an extra dimension to 'norms' so the broadcasting works correctly during division
                    normalized_last_three_components = last_three_components / norms[:, :, np.newaxis]

                    # Set the normalized direction vectors to NaN where they were originally NaN
                    normalized_last_three_components[invalid_indices] = np.nan

                    # Concatenate the first three components with the normalized last three components to get the final result
                    interpolated_coordinates = np.concatenate((first_three_components, normalized_last_three_components), axis=2)


                    
                    # optional
                    # # Find the indices where values are nan
                    # nan_indices = np.argwhere(np.isnan(interpolated_coordinates))
                    # # Perform nearest interpolation only at the nan locations
                    # interpolated_coordinates[nan_indices[:,0], nan_indices[:,1]] = griddata(non_zero_2d_indices, non_zero_coordinates, (dense_grid[0][nan_indices[:,0], nan_indices[:,1]], dense_grid[1][nan_indices[:,0], nan_indices[:,1]]), method='nearest')
                    
                    fig, ax = plt.subplots(figsize=(30, 30))
                    plt.imshow(interpolated_coordinates[:,:,4].T)
                    plt.colorbar()
                    plt.savefig("sample_points_plot_inter.png")




                    samples_forward = 25
                    samples_backward = 2
                    total_samples = samples_forward + samples_backward + 1 # there's one for the 0th layer

                    # Shape is layer, X, Y, position to sample from original volume
                    sample_positions = np.zeros((total_samples, interpolated_coordinates.shape[0], interpolated_coordinates.shape[1], 3))
                    for i in range(-samples_backward, samples_forward):
                        print("i", i)
                        # negative sign because the stuff seems to be reversed
                        sample_positions[i + samples_backward, :, :, :] = interpolated_coordinates[:, :, :3] + -i * interpolated_coordinates[:, :, 3:]


                    # Perform the sampling
                    x = np.arange(volume_no_threshold.shape[0])
                    y = np.arange(volume_no_threshold.shape[1])
                    z = np.arange(volume_no_threshold.shape[2])

                    volume_no_threshold[0,0,0] = 0

                    interpolator = RegularGridInterpolator((x, y, z), volume_no_threshold)

                    sample_positions = np.nan_to_num(sample_positions + center)

                    sample_positions = np.clip(sample_positions, 0, np.array([x.max(), y.max(), z.max()]))

                    # Reshape the points to a 2D array
                    points_reshaped = sample_positions.reshape(-1, 3)

                    # Perform the interpolation
                    result_reshaped = interpolator(points_reshaped)

                    # Reshape the result back to the original shape
                    result = result_reshaped.reshape(sample_positions.shape[:-1])

                    # volume_p = result[6:14, :, :].astype(np.uint8)
                    # # volume_p = np.transpose(volume_p, (2, 1, 0))
                    # volume_size = volume_p.shape

                    # fig, ax = plt.subplots(figsize=(30, 30))
                    # plt.imshow(result[6, :, :])
                    # plt.colorbar()
                    # plt.savefig("hhh.png")

                    t = int(time.time())
                    output_folder = f"./segmentations/{t}"
                    os.makedirs(output_folder, exist_ok=True)
                    os.makedirs(os.path.join(output_folder, "volume"), exist_ok=True)

                    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
                        json.dump({
                            "timestamp": t,
                            "url": request.url,
                            "area_um2": np.count_nonzero(result[10, :, :]) * 8 # each pixel is 8um
                        }, f)
                        
                    for i in range(result.shape[0]):
                        # fig, ax = plt.subplots(figsize=(30, 30))
                        # plt.imshow(result[i, :, :])
                        # plt.colorbar()
                        # plt.savefig(f"hhh{i}.png")                        
                        imageio.imwrite(os.path.join(os.path.join(output_folder, "volume"), f'{i}.png'), result[i, :, :].astype(np.uint8))

                    mask = np.sum(result[:total_samples-2, :, :], axis=0) != 0
                    imageio.imwrite(os.path.join(output_folder, f'mask.png'), (mask * 255).astype(np.uint8))

                    # # Set these vectors to zero
                    # volume_p[invalid_indices] = 0
                    # volume_p = (mask_of_vectors_to_keep * 254).astype(np.uint8)
                    # volume_p = (x_avg * 254).astype(np.uint8)
                    # volume_size = x_avg.shape
                    
                    # Show the surface over the original isolated segment.
                    volume_p = (volume_no_threshold / 3).astype(np.uint8)
                    # volume_p[sample_positions[samples_backward, :, :, :]] = 255
                        
                    for p in sample_positions[0, :, :, :].reshape(-1, 3).astype(np.int32):
                        volume_p[p[0], p[1], p[2]] = 210

                    for p in sample_positions[samples_backward, :, :, :].reshape(-1, 3).astype(np.int32):
                        volume_p[p[0], p[1], p[2]] = 230
                        
                    for p in sample_positions[total_samples - 2, :, :, :].reshape(-1, 3).astype(np.int32):
                        volume_p[p[0], p[1], p[2]] = 250

                # I don't know why this is required, think about this
                # Get the pixel data as a bytes object and add it to the raw_data bytearray
                raw_data = bytearray(volume_size[0] // (lod_downsample) * volume_size[1] // (lod_downsample) * (volume_size[2] // lod_downsample))
                for i in range(volume_size[2]):
                    layer_data = volume_p[:, :, i].tobytes()
                    raw_data[i * volume_size[0] * volume_size[1] : (i + 1) * volume_size[0] * volume_size[1]] = layer_data
                
                volume = raw_data
            
            else:
                print("uh")
            
            print(config[filename]["dimensions"], "foo")

        # volume and volume_size must be assigned by now
        assert volume and volume_size

        # Create a new bytearray and add the size as uint32 values at the start
        volume_with_shape = bytearray()
        volume_with_shape.extend(struct.pack('<III', volume_size[0], volume_size[1], volume_size[2]))
        volume_with_shape.extend(volume)

        # Convert the bytearray to bytes if needed
        volume = bytes(volume_with_shape)
        print("Vol_shape")
        
        # Create a binary stream to store the volume data
        binary_stream = BytesIO()
        binary_stream.write(volume)
        binary_stream.seek(0)

        print("size expected:", volume_size[0] * volume_size[1] * volume_size[2], "stream nbytes:", binary_stream.getbuffer().nbytes)

        # Serve the .raw file as a static file
        return send_file(binary_stream, download_name="volume.raw", as_attachment=True)

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return f"Error: {str(e)}", 500


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
app.logger.addHandler(console_handler)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
