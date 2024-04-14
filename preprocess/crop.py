'''
Crops a 20x20 top-left part of the image to test fake classification of models.
'''

import os
import cv2
from keras.preprocessing import image

src_data_dir = '../../data/caltech_20/test/'
dest_data_dir = '../../data/caltech_20/cropped/test/'

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

        # Load the image
        image = cv2.imread(img_path)

        # Define the coordinates of the region to be cropped (top-left and bottom-right corners)
        x1, y1 = 0, 0  # Top-left corner
        x2, y2 = 20, 20  # Bottom-right corner

        # Crop the region of interest from the image
        cropped_image = image[y1:y2, x1:x2]

        dest_img_path = os.path.join(dest_data_dir, folder, img)

        # Save the cropped image as a new file
        cv2.imwrite(dest_img_path, cropped_image)

        print('Cropped: ', dest_img_path)
