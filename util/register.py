import numpy as np
from scipy.ndimage import shift
from skimage.metrics import structural_similarity as ssim
import imageio
import os
import tqdm
from PIL import Image
import cv2
import time

def read_volume(folder_path):
    file_names = sorted(os.listdir(folder_path))
    slices = [imageio.imread(os.path.join(folder_path, file_name)) for file_name in file_names]
    volume = np.stack(slices, axis=-1)
    return volume

def read_mask(mask_file_path, volume_shape):
    mask = imageio.imread(mask_file_path)
    mask = mask.astype(bool)  # Convert to boolean mask if not already
    mask_volume = np.repeat(mask[:, :, np.newaxis], volume_shape[-1], axis=-1)
    return mask_volume

def save_volume(volume, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, slice in enumerate(np.moveaxis(volume, -1, 0)):
        imageio.imsave(os.path.join(folder_path, f"image_{i}.png"), slice)

def save_image(image, file_path):
    imageio.imsave(file_path, image)


def stitch_images(mask1, mask2, offset):
    # load the mask images (assuming they are grayscale images)
    # mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    # mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # define your offset (dy, dx)
    dy, dx = offset

    # get the shape of each mask
    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # compute the size of the new image
    new_h = max(h1 - min(0, dy), h2 + max(0, dy))
    new_w = max(w1 - min(0, dx), w2 + max(0, dx))

    # create a new array with a size large enough to store both images
    result = np.zeros((new_h, new_w), dtype=np.uint8)

    # compute where to place each mask
    start_y1 = max(0, -dy)
    start_x1 = max(0, -dx)
    start_y2 = max(0, dy)
    start_x2 = max(0, dx)

    # place mask1
    result[start_y1:start_y1+h1, start_x1:start_x1+w1] = mask1

    # place mask2 with the given offset
    result[start_y2:start_y2+h2, start_x2:start_x2+w2] = mask2

    return result

def stitch_images_or(mask1, mask2, offset):
    # # load the mask images (assuming they are grayscale images)
    # mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    # mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # define your offset (dy, dx)
    dy, dx = offset[0], offset[1]

    # get the shape of each mask
    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # compute the size of the new image
    new_h = max(h1 - min(0, dy), h2 + max(0, dy))
    new_w = max(w1 - min(0, dx), w2 + max(0, dx))

    # create a new array with a size large enough to store both images
    result1 = np.zeros((new_h, new_w), dtype=np.uint8)
    result2 = np.zeros((new_h, new_w), dtype=np.uint8)

    # compute where to place each mask
    start_y1 = max(0, -dy)
    start_x1 = max(0, -dx)
    start_y2 = max(0, dy)
    start_x2 = max(0, dx)

    # place mask1
    result1[start_y1:start_y1+h1, start_x1:start_x1+w1] = mask1

    # place mask2 with the given offset
    result2[start_y2:start_y2+h2, start_x2:start_x2+w2] = mask2

    # perform a logical OR operation
    result = cv2.bitwise_or(result1, result2)

    return result

def stitch_images_blit(image1, image2, offset):
    # load the grayscale images
    # image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # define your offset (dy, dx)
    dy, dx = offset

    # get the shape of each image
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # compute the size of the new image
    new_h = max(h1 - min(0, dy), h2 + max(0, dy))
    new_w = max(w1 - min(0, dx), w2 + max(0, dx))

    # create a new array with a size large enough to store both images
    result = np.zeros((new_h, new_w), dtype=np.uint8)

    # compute where to place each image
    start_y1 = max(0, -dy)
    start_x1 = max(0, -dx)
    start_y2 = max(0, dy)
    start_x2 = max(0, dx)

    # place image1
    result[start_y1:start_y1+h1, start_x1:start_x1+w1] = image1

    # create mask where image2 has non-zero values
    mask2 = image2 > 0

    # place image2 at the offset position, only where image2 has non-zero values
    result[start_y2:start_y2+h2, start_x2:start_x2+w2][mask2] = image2[mask2]

    return result


def stitch_images_blit_fade(image1, image2, offset):
    # load the grayscale images
    # image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # define your offset (dy, dx)
    dy, dx = offset

    # get the shape of each image
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # compute the size of the new image
    new_h = max(h1 - min(0, dy), h2 + max(0, dy))
    new_w = max(w1 - min(0, dx), w2 + max(0, dx))

    # create new arrays with a size large enough to store both images
    result = np.zeros((new_h, new_w), dtype=np.uint8)
    result1 = np.zeros((new_h, new_w), dtype=np.uint8)
    result2 = np.zeros((new_h, new_w), dtype=np.uint8)

    # compute where to place each image
    start_y1 = max(0, -dy)
    start_x1 = max(0, -dx)
    start_y2 = max(0, dy)
    start_x2 = max(0, dx)

    # place image1
    result1[start_y1:start_y1+h1, start_x1:start_x1+w1] = image1

    # place image2
    result2[start_y2:start_y2+h2, start_x2:start_x2+w2] = image2

    # create mask where both images have non-zero values
    overlap_mask = (result1 > 0) & (result2 > 0)

    # calculate distance transform for fading effect
    dist_transform = cv2.distanceTransform(result2.astype(np.uint8), cv2.DIST_L2, 3)

    # normalize distance transform (this will create the fading effect, increase 100 for a wider fading area)
    cv2.normalize(dist_transform, dist_transform, 0, 10, cv2.NORM_MINMAX)

    # apply fading effect to the overlapping area of image2
    result2_fade = cv2.multiply(result2.astype(np.float32), dist_transform / 10)

    # place the faded image2 into the final result only in the overlapping area
    result[overlap_mask] = result2_fade[overlap_mask]

    # place non-overlapping parts of image1 and image2 into the final result
    result[np.logical_and(result1 > 0, ~overlap_mask)] = result1[np.logical_and(result1 > 0, ~overlap_mask)]
    result[np.logical_and(result2 > 0, ~overlap_mask)] = result2[np.logical_and(result2 > 0, ~overlap_mask)]

    return result

def stitch_images_average(image1, image2, offset):
    # load the grayscale images
    # image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # define your offset (dy, dx)
    dy, dx = offset

    # get the shape of each image
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # compute the size of the new image
    new_h = max(h1 - min(0, dy), h2 + max(0, dy))
    new_w = max(w1 - min(0, dx), w2 + max(0, dx))

    # create a new array with a size large enough to store both images
    result = np.zeros((new_h, new_w), dtype=np.uint8)

    # compute where to place each image
    start_y1 = max(0, -dy)
    start_x1 = max(0, -dx)
    start_y2 = max(0, dy)
    start_x2 = max(0, dx)

    # place image1
    result[start_y1:start_y1+h1, start_x1:start_x1+w1] = image1

    # create mask where image2 has non-zero values
    mask2 = image2 > 0

    # identify where both images have non-zero values
    overlap = ((result[start_y2:start_y2+h2, start_x2:start_x2+w2] > 0) & mask2)

    # calculate the average of overlapping non-zero values
    result[start_y2:start_y2+h2, start_x2:start_x2+w2][overlap] = \
        ((result[start_y2:start_y2+h2, start_x2:start_x2+w2][overlap] + image2[overlap]) / 2).astype(np.uint8)

    # place image2 at the offset position, only where image2 has non-zero values and there's no overlap
    result[start_y2:start_y2+h2, start_x2:start_x2+w2][mask2 & ~overlap] = image2[mask2 & ~overlap]

    return result

# Rest of the code remains the same as before
def brute_force_registration_3d(volume1_folder_path, volume2_folder_path, mask_file_path, mask2_file_path, output_folder_path, overlap_percent, min_distance=60):
    full_volume1 = read_volume(volume1_folder_path)
    full_volume2 = read_volume(volume2_folder_path)
    volume1 = full_volume1[:,:,7:23]
    volume2 = full_volume2[:,:,7:23]
    mask = read_mask(mask_file_path, volume1.shape)
    mask2 = read_mask(mask2_file_path, volume2.shape)
    
    dims = volume1.shape
    max_shift_x = int(dims[0] * overlap_percent)
    max_shift_y = int(dims[1] * overlap_percent)
    
    best_shift = (0, 0, 0)
    best_ssim = -1
    # shifts = [(dx, dy, 0) for dx in range(-max_shift_x, max_shift_x + 1) for dy in range(-max_shift_y, max_shift_y + 1)]
    # for downsample_factor in [16, 4, 1]:
    for downsample_factor in [8]:
        print ("Running with factor", downsample_factor)
        shifts = []
        if best_ssim == -1: # first run
            shifts = [(dx, dy, 0) for dx in range(-max_shift_x, max_shift_x + 1, downsample_factor) for dy in range(-max_shift_y, max_shift_y + 1, downsample_factor)]
        else:
            shifts = [(dx, dy, 0) for dx in range(best_shift[0] - 10 * downsample_factor, best_shift[0] + 10 * downsample_factor, downsample_factor) for dy in range(best_shift[1] - 10 * downsample_factor, best_shift[1] + 10 * downsample_factor + 1, downsample_factor)]
        
        for shift_xyz in tqdm.tqdm(shifts):
            if np.sqrt(shift_xyz[0] ** 2 + shift_xyz[1] ** 2) < min_distance: # skip close alignments
                continue
            shifted_volume2 = shift(volume2, shift_xyz)
            shifted_mask2 = shift(mask, shift_xyz)  # Shift the mask as well

            # Crop if necessary
            if shifted_volume2.shape[0] > dims[0]:
                shifted_volume2 = shifted_volume2[:dims[0], :, :]
            if shifted_volume2.shape[1] > dims[1]:
                shifted_volume2 = shifted_volume2[:, :dims[1], :]
            if shifted_volume2.shape[2] > dims[2]:
                shifted_volume2 = shifted_volume2[:, :, :dims[2]]

            # Do the same for the mask
            if shifted_mask2.shape[0] > dims[0]:
                shifted_mask2 = shifted_mask2[:dims[0], :, :]
            if shifted_mask2.shape[1] > dims[1]:
                shifted_mask2 = shifted_mask2[:, :dims[1], :]
            if shifted_mask2.shape[2] > dims[2]:
                shifted_mask2 = shifted_mask2[:, :, :dims[2]]
                
            pad_x = dims[0] - shifted_volume2.shape[0]
            pad_y = dims[1] - shifted_volume2.shape[1]
            pad_z = dims[2] - shifted_volume2.shape[2]
            shifted_volume2 = np.pad(shifted_volume2, ((0, pad_x), (0, pad_y), (0, pad_z)))

            pad_x = dims[0] - shifted_mask2.shape[0]
            pad_y = dims[1] - shifted_mask2.shape[1]
            pad_z = dims[2] - shifted_mask2.shape[2]
            shifted_mask2 = np.pad(shifted_mask2, ((0, pad_x), (0, pad_y), (0, pad_z)))  # Pad the mask as well

            combined_mask = np.logical_and(mask, shifted_mask2)
            
            ssim_map, _ = ssim(volume1, shifted_volume2, full=True)
            masked_ssim_map = ssim_map * combined_mask
            score = np.mean(masked_ssim_map[combined_mask])  # Average over the mask

            if score > best_ssim:
                best_ssim = score
                best_shift = shift_xyz
                print(score, shift_xyz)

    # fix this
    volume1 = full_volume1
    volume2 = full_volume2
    mask = read_mask(mask_file_path, volume1.shape)
    mask2 = read_mask(mask2_file_path, volume2.shape)
    
    dims = volume1.shape

    registered_volume2 = shift(full_volume2, best_shift)
    registered_mask2 = shift(mask2, best_shift)  # Shift the mask

    dims = full_volume1.shape
    # Define new dimensions based on the applied shift and original volumes
    new_dims = (
        max(dims[0], registered_volume2.shape[0] + abs(best_shift[0])),
        max(dims[1], registered_volume2.shape[1] + abs(best_shift[1])),
        max(dims[2], registered_volume2.shape[2] + abs(best_shift[2]))
    )

    # Resize volumes and masks to the new dimensions
    full_volume1 = np.pad(full_volume1, ((0, new_dims[0] - dims[0]), (0, new_dims[1] - dims[1]), (0, new_dims[2] - dims[2])))
    mask1 = np.pad(mask, ((0, new_dims[0] - dims[0]), (0, new_dims[1] - dims[1]), (0, new_dims[2] - dims[2])))
    registered_volume2 = np.pad(registered_volume2, ((0, new_dims[0] - registered_volume2.shape[0]), (0, new_dims[1] - registered_volume2.shape[1]), (0, new_dims[2] - registered_volume2.shape[2])))
    registered_mask2 = np.pad(registered_mask2, ((0, new_dims[0] - registered_mask2.shape[0]), (0, new_dims[1] - registered_mask2.shape[1]), (0, new_dims[2] - registered_mask2.shape[2])))

    os.makedirs(output_folder_path)
    os.makedirs(os.path.join(output_folder_path, "volume"))

    # Combine masks
    save_image(stitch_images_or(mask1[:,:,0], mask2[:,:,0], best_shift) * 255, os.path.join(output_folder_path, 'mask.png'))

    # Combine images
    for i in range(volume1.shape[2]):
        save_image(stitch_images_blit(full_volume1[:,:,i], full_volume2[:,:,i], (best_shift[0], best_shift[1])), os.path.join(output_folder_path, f"volume/{i}.png"))


    # save_volume(combined_volume.astype(np.uint8), os.path.join(output_folder_path, 'combined_volume'))
    # save_volume(combined_mask.astype(np.uint8), os.path.join(output_folder_path, 'combined_mask'))

if __name__ == '__main__':
    # Provide your folder paths and overlap percentage
    volume1_folder_path = "./segmentations/1686740127/volume" # super simple first segments
    volume2_folder_path = "./segmentations/1686740413/volume"
    mask_file_path = "./segmentations/1686740127/mask.png"
    mask2_file_path = "./segmentations/1686740413/mask.png"
    output_folder_path = f"./registrations/{int(time.time())}"
    overlap_percent = 0.1

    brute_force_registration_3d(volume1_folder_path, volume2_folder_path, mask_file_path, mask2_file_path, output_folder_path, overlap_percent=0.9)
