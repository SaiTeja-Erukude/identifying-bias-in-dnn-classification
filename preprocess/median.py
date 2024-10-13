'''
Attempt to remove noise
Applies Median filtering to all images in the source directory and then saves them to the destination directory.
'''

import os
import cv2
import numpy as np
 
# source and destination data dir
src_data_dir = '../../data/natural_images/original/cropped/val/'
dest_data_dir = '../../data/natural_images/original/cropped/median/val/'

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


    for im in images:

        # form the image path
        img_path = os.path.join(src_data_dir, folder, im)

        # now we will be loading the image and converting it to grayscale
        image = cv2.imread(img_path)
               
        # apply median filtering
        median = cv2.medianBlur(image, 5)

        dest_img_path = os.path.join(dest_data_dir, folder, im)
        
        cv2.imwrite(dest_img_path, median)

        print('Saved: ', dest_img_path)
