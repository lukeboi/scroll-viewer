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

from converttoraw import convert_tif_stack_to_raw

app = Flask(__name__)
CORS(app, supports_credentials=True)

json_file = "config.json"
import traceback

# global status message for the user
server_status = "Server started"

@app.errorhandler(Exception)
def handle_exception(e):
    #Use for stack trace
    return traceback.format_exc()


def get_volume_from_tif_stack(src, origin, size, depth="8", extension="tif", threshold=0):
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
    raw_data = bytearray(size[0] * size[1] * size[2])

    # Iterate over the TIF files, converting them to R8 format and adding them to the raw_data bytearray
    print(tif_files)
    for i, tif_file in enumerate(tif_files[origin[2]:origin[2] + size[2]]):
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

        print(np.array(image).shape)
        # Convert the image to grayscale (single channel, 8 bits)
        # gray_image = image.convert("L")
        # gray_image.show()

        # print(image.shape)
        # print(image.dtype)

        # Crop the image
        # gray_image = gray_image.crop((origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]))
        image = image[origin[0]:origin[0] + size[0], origin[1]:origin[1] + size[1]]

        # print(np.array(image).mean())
        
        if depth == "16":
            image = (image * (255 / 65535)).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)
        # image = image.astype(np.uint8)

        print(np.array(image).shape)

        # Apply threshold
        # threshold = 0
        image[image < threshold] = 0
        image[image >= threshold] = ((image[image >= threshold] - threshold) / (255 - threshold)) * 255


        # Get the pixel data as a bytes object and add it to the raw_data bytearray
        pixel_data = image.tobytes()
        raw_data[i * size[0] * size[1] : (i + 1) * size[0] * size[1]] = pixel_data

        print(i)

        server_status = f"Loading: {i}/{len(tif_files[origin[2]:origin[2] + size[2]])}"

    server_status = "Done Loading!"

    # raw_data = raw_data.transpose(raw_data, (2, 1, 0))

    return raw_data

# funny little heartbeat
heartbeat_counter = 0
heartbeat_threshold = 10

@app.route('/heartbeat', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_heartbeat():
    global heartbeat_counter
    heart = "< 3" if heartbeat_counter % heartbeat_threshold == 0 else "<3"
    heartbeat_counter += 1

    response = make_response(heart + "<br>" + server_status, 200)
    response.mimetype = "text/plain"

    return response

@cross_origin(supports_credentials=True)
@app.route('/volume_metadata', methods=['GET'])
def get_volume_metadata():
    return send_file("config.json")

@app.route('/volume', methods=['GET'])
def volume():
    try:
        filename = request.args.get('filename') # Requested filename for lookup in config.json

        # Size and origin of request, in original resoultion.
        size = [int(x) for x in request.args.get('size').split(',')]
        origin = [int(x) for x in request.args.get('origin').split(',')]

        # Load the config
        config = None
        with open(json_file, 'r') as f:
            config = json.load(f)

        # Volume information
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

            # for i in range(volume.shape[0]):
            #     for j in range(volume.shape[1]):
            #         for k in range(volume.shape[2]):
            #             volume[i, :, :] = 128
            # print(volume)
            volume = np.transpose(volume, (2, 1, 0))
            volume = volume.tobytes()

            volume_size = size
            # print(volume)
            # volume = np.asfortranarray(volume)

        else:
            print(config)
            # Load that volume
            if filename in config:
                # load from a tif stack
                print("loading")
                volume_p = get_volume_from_tif_stack(config[filename]["src"], origin, size, config[filename]["depth"], config[filename]["extension"], int(request.args.get('threshold')))
                volume_size = size
                # print("dunn")

                # load from raw file
                # volume_p = np.fromfile(config[filename]["src"], dtype=np.uint8)

                # # crop
                # # volume = volume_precrop[0:size[0], 0:size[1], 0:size[2]]
                # volume_p[500:560:1] = 255 # draw line on x axis
                # volume_p[500 * 560:560 * 560:560] = 255 # draw line on y axis
                # volume_p[400 * 560 * 560::560 * 560] = 255 # draw line on z axis

                # # # draw cube in mesh
                # # for y in range(100):
                # #     for z in range(100):
                # #         volume_p[500:560:1] = 255 # draw line on x axis
                # volume_p = np.reshape(volume_p, (560, 560, 477), order='C')
                # volume_p = np.transpose(volume_p, (2, 0, 1))
                # volume_p[10:200, 10:20, 10:20] = 128
                # volume_p = np.transpose(volume_p, (2, 0, 1))


                # # Create meshgrid arrays for x, y, and z indices
                # x_range = np.arange(0, 550)
                # y_range = np.arange(0, 550)
                # z_range = np.arange(0, 450)

                # x_indices, y_indices, z_indices = np.meshgrid(x_range, y_range, z_range, indexing='ij')

                # # Calculate 1D indices for the original volume and crop volume
                # original_indices = z_indices * 560 * 560 + y_indices * 560 + x_indices
                # print(original_indices.ravel(), original_indices.shape)

                # volume = volume_p[original_indices.ravel()]

                # # # reshape into 3d
                # # volume = np.reshape(volume, config[filename]["dimensions"])
                # # volume = np.transpose(volume, (0, 2, 1))
                
                volume = volume_p
            
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
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
