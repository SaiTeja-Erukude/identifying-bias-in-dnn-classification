'''
Scrambles the input images and saves in the destination dir.
Selects tiles from the input image and saves at random positions in the new image.
'''

import os
import cv2
import random
import numpy as np
 
# source and destination data dir
src_data_dir = '../../data/natural_images/original/train/'
dest_data_dir = '../../data/natural_images/original/scrambled/tile_size32/train/'

# height and width of input images
height = width = 224

# tile size
tile_size = 32

# Get a list of folders in the directory
folders = []
for folder in os.listdir(src_data_dir):
    if os.path.isdir(os.path.join(src_data_dir, folder)):
        folders.append(folder)

for folder in folders:

    # Get the list of images in the folder
    images = []
    for im in os.listdir(os.path.join(src_data_dir, folder)):
        if im.endswith('.jpg') or im.endswith('.png') or im.endswith('.JPEG') or im.endswith('.jpeg'):
            images.append(im)

    for img in images:

        # form the image path
        img_path = os.path.join(src_data_dir, folder, img)

        # Read the image
        input_img = cv2.imread(img_path)

        # Resize the image to the target size (224x224)
        input_img = cv2.resize(input_img, (height, width))
        
        # Initialize empty list to store tile images
        tiles = []
        
        # Extract square tiles from the input image
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = input_img[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)

        # Shuffle the order of the extracted tiles
        random.shuffle(tiles)

        # Create a new blank image to place the shuffled tiles
        output_image = np.zeros_like(input_img)

        # Place shuffled tiles back into the new image
        index = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                output_image[y:y+tile_size, x:x+tile_size] = tiles[index]
                index += 1

        # save the output image       
        dest_img_path = os.path.join(dest_data_dir, folder, img)
        cv2.imwrite(dest_img_path, output_image)
        print('Saved: ', dest_img_path)