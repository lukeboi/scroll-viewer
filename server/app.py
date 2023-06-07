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

def sobel_filter_3d(input, chunks=4, overlap=3):
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
        if i != chunks - 1:  # Not the last chunk
            G = G[:, :, :-chunk_overlap, :, :]

        # Move the result back to CPU and add it to the list
        results.append(G.cpu())

        # Free memory of intermediate variables
        del G_x, G_y, G_z, chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate the results along the depth dimension
    result = torch.cat(results, dim=2)

    return result

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

        # Apply threshold
        # threshold = 0
        image[image < threshold] = 0
        image[image >= threshold] = ((image[image >= threshold] - threshold) / (255 - threshold)) * 255

        print(i)

        data[:, :, i] = image

        server_status = f"Loading: {i}/{len(tif_files[origin[2]:origin[2] + size[2]])}"

    server_status = "Done Loading!"

    # raw_data = raw_data.transpose(raw_data, (2, 1, 0))

    return data, shape

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

    return response

# @cross_origin(supports_credentials=True)
@app.route('/volume_metadata', methods=['GET'])
def get_volume_metadata():
    global json_file
    return send_file(json_file)

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
                print("loading")
                volume_p, volume_size = get_volume_from_tif_stack(config[filename]["src"], origin, size,
                                                     lod_downsample=1,
                                                     depth=config[filename]["depth"],
                                                     extension=config[filename]["extension"],
                                                     threshold=int(request.args.get('threshold')))
                
                if apply_sobel:
                    volume_p = sobel_filter_3d(torch.from_numpy(volume_p.astype(np.float32)).unsqueeze(0).unsqueeze(0)).numpy().astype(np.uint8).squeeze(0).squeeze(0)
                
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
        print(e)
        return f"Error: {str(e)}", 500


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
app.logger.addHandler(console_handler)

if __name__ == '__main__':
    app.run(debug=True)
