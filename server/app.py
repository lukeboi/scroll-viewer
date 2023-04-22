import numpy as np
from io import BytesIO
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/random_volume', methods=['GET'])
def random_volume():
    try:
        size = request.args.get('size')
        dimensions = [int(x) for x in size.split(',')]

        if len(dimensions) != 3:
            return "Invalid input. Please provide a 3D volume size in the format 'width,height,depth'.", 400

        width, height, depth = dimensions

        # Generate the random 3D volume with values in the range 0 to 255 (8-bit)
        random_volume = np.random.randint(0, 256, size=(width, height, depth), dtype=np.uint8)

        # Create a binary stream to store the volume data
        binary_stream = BytesIO()
        binary_stream.write(random_volume.tobytes())
        binary_stream.seek(0)

        # Serve the .raw file as a static file
        return send_file(binary_stream, download_name="random_volume.raw", as_attachment=True)

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/load_subvolume', methods=['GET'])
def load_subvolume():
    try:
        json_file = request.args.get('json_file')
        volume_filename = request.args.get('volume_filename')

        with open(json_file, 'r') as f:
            config = json.load(f)

        if volume_filename is None:
            volume_filename = config.get('volume_filename')

        if not os.path.exists(volume_filename):
            return f"Error: Volume file not found at '{volume_filename}'", 404

        subvolume_coords = config.get('subvolume_coords')
        if subvolume_coords is None:
            return "Error: 'subvolume_coords' not found in the JSON configuration.", 400

        x1, y1, z1, x2, y2, z2 = subvolume_coords

        volume_data = np.fromfile(volume_filename, dtype=np.uint8)
        volume_shape = config.get('volume_shape', (256, 256, 256))  # Assuming a default shape
        volume = np.reshape(volume_data, volume_shape)

        subvolume = volume[x1:x2, y1:y2, z1:z2]

        # Create a binary stream to store the subvolume data
        binary_stream = BytesIO()
        binary_stream.write(subvolume.tobytes())
        binary_stream.seek(0)

        # Serve the .raw file as a static file
        return send_file(binary_stream, download_name="subvolume.raw", as_attachment=True)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
