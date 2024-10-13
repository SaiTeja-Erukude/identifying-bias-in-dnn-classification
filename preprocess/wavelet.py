'''
Applies Wavelet transform to all images in the source directory and then saves them to the destination directory.

pywt.families()
['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
pywt.families(short=False)
['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)', 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet', 'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets']

'''

import os
import cv2
import pywt
import numpy as np
 
# source and destination data dir
src_data_dir = '../../data/natural_images/original/cropped/median/val/'
dest_data_dir = '../../data/natural_images/wavelet/haar/cropped/median/val/'

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
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
              
        # Perform 2D Discrete Wavelet Transform (DWT)
        # Haar Wavelet - haar
        # Daubechies Orthogonal Wavelet - db2
        coeffs = pywt.dwt2(image, 'haar')

        # Extract approximation, horizontal, vertical, and diagonal coefficients
        cA, (cH, cV, cD) = coeffs

        # Concatenate the coefficients into a single array
        # coefficients_image = np.concatenate((cA, cH, cV, cD), axis=1)        

        dest_img_path = os.path.join(dest_data_dir, folder, im)
        
        cv2.imwrite(dest_img_path, cA.astype(np.uint8))

        print('Saved: ', dest_img_path)
