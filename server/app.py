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
                    # volume_p = np.where(labels3d == target_value, volume_no_threshold, 0)
                    volume_p = np.where(labels3d == target_value, volume_p, 0)

                    # cpu gaussian fliter. lol.
                    volume_p = gaussian_filter(volume_p * 255, sigma=6).astype(np.uint8) > 0
                    sobel_output = sobel_filter_3d(torch.from_numpy((volume_p).astype(np.float32)).unsqueeze(0).unsqueeze(0), return_vectors=True)
                    volume_p = sobel_output[0].numpy().astype(np.uint8).squeeze(0).squeeze(0)
                    sobel_vectors = sobel_output[1].squeeze(0).squeeze(0)

                    volume_p = (volume_p * 254).astype(np.uint8)

                
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
