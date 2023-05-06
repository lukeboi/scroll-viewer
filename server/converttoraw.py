import os
from PIL import Image

input_directory = "../campfire/rec"
output_file = "output.raw"

def convert_tif_stack_to_raw(input_directory, output_file):
    # Get the list of TIF files in the input directory
    tif_files = [f for f in os.listdir(input_directory) if f.lower().endswith(".tif")]

    # Sort the TIF files alphabetically to maintain the correct order
    tif_files.sort()

    # Open the first image to get its dimensions
    first_image = Image.open(os.path.join(input_directory, tif_files[0]))
    width, height = first_image.size
    depth = len(tif_files)

    print("Dimensions:", width, height, depth)

    # Create an empty bytearray to store the .raw data
    raw_data = bytearray(width * height * len(tif_files))

    # Iterate over the TIF files, converting them to R8 format and adding them to the raw_data bytearray
    for i, tif_file in enumerate(tif_files):
        image_path = os.path.join(input_directory, tif_file)
        image = Image.open(image_path)

        # Convert the image to grayscale (single channel, 8 bits)
        gray_image = image.convert("L")

        # Get the pixel data as a bytes object and add it to the raw_data bytearray
        pixel_data = gray_image.tobytes()
        raw_data[i * width * height : (i + 1) * width * height] = pixel_data

    return raw_data

if __name__ == "__main__":
    raw_data = convert_tif_stack_to_raw(input_directory, output_file)

    # Write the .raw data to the output file
    with open(output_file, "wb") as f:
        f.write(raw_data)
        print("Done!")
