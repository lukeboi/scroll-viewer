import numpy as np
from io import BytesIO
from flask import Flask, request, send_file
from flask_cors import CORS
import json
import traceback
import os
from PIL import Image
import tifffile as tiff

from converttoraw import convert_tif_stack_to_raw

app = Flask(__name__)
CORS(app)

json_file = "config.json"
import traceback

# global status message for the user
server_status = "yoinky sploinky"

@app.errorhandler(Exception)
def handle_exception(e):
    #Use for stack trace
    return traceback.format_exc()


def get_volume_from_tif_stack(src, origin, size):
    # Get the list of TIF files in the input directory
    tif_files = [f for f in os.listdir(src) if f.lower().endswith(".tif")]

    # Sort the TIF files alphabetically to maintain the correct order
    tif_files.sort()

    # Open the first image to get its dimensions
    # first_image = Image.open(os.path.join(src, tif_files[0]))
    # width, height = first_image.size
    # depth = len(tif_files)

    # print("Dimensions:", width, height, depth)

    # Create an empty bytearray to store the .raw data
    # raw_data = bytearray(width * height * len(tif_files))
    raw_data = bytearray(size[0] * size[1] * size[2])

    # Iterate over the TIF files, converting them to R8 format and adding them to the raw_data bytearray
    print(tif_files)
    for i, tif_file in enumerate(tif_files[origin[2]:origin[2] + size[2]]):
        image_path = os.path.join(src, tif_file)
        # image = Image.open(image_path)
        image = tiff.imread(image_path)

        # Convert the image to grayscale (single channel, 8 bits)
        # gray_image = image.convert("L")
        # gray_image.show()

        print(image.shape)
        print(image.dtype)

        # Crop the image
        # gray_image = gray_image.crop((origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]))
        image = image[origin[0]:origin[0] + size[0], origin[1]:origin[1] + size[1]]

        print(np.array(image).mean())
        
        image = (image * (255 / 65535)).astype(np.uint8)

        print(np.array(image).mean())

        # Get the pixel data as a bytes object and add it to the raw_data bytearray
        pixel_data = image.tobytes()
        raw_data[i * size[0] * size[1] : (i + 1) * size[0] * size[1]] = pixel_data

        print(i)

    return raw_data


@app.route('/volume_metadata', methods=['GET'])
def get_volume_metadata():
    return send_file("config.json")

@app.route('/heartbeat', methods=['GET'])
def get_heartbeat():
    return server_status

@app.route('/volume', methods=['GET'])
def volume():
    try:
        s = request.args.get('size')
        o = request.args.get('origin')
        filename = request.args.get('filename')
        size = [int(x) for x in s.split(',')]
        origin = [int(x) for x in o.split(',')]

        print(size)

        config = None
        with open(json_file, 'r') as f:
            config = json.load(f)

        volume = None
        print("request")

        if filename == "random_volume":
            # Generate the random 3D volume with values in the range 0 to 255 (8-bit)
            volume = np.random.randint(0, 256, size=(size[0], size[1], size[2]), dtype=np.uint8)
            volume = volume.tobytes()
        else:
            print(config)
            # Load that volume
            if filename in config:
                # load from a tif stack
                print("loading")
                volume_p = get_volume_from_tif_stack(config[filename]["src"], origin, size)
                print("dunn")

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

        # Create a binary stream to store the volume data
        binary_stream = BytesIO()
        binary_stream.write(volume)
        binary_stream.seek(0)

        # Serve the .raw file as a static file
        return send_file(binary_stream, download_name="volume.raw", as_attachment=True)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
