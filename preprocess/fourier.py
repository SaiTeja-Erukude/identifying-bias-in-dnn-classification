'''
Applies Fourier transform to all images in the source directory and then saves them to the destination directory.
'''

import os
import cv2
import numpy as np
 
# source and destination data dir
src_data_dir = ''
dest_data_dir = ''

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
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute the discrete Fourier Transform of the image
        fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Shift the zero-frequency component to the center of the spectrum
        fourier_shift = np.fft.fftshift(fourier)

        # calculate the magnitude of the Fourier Transform
        magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
        
        # Scale the magnitude for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        dest_img_path = os.path.join(dest_data_dir, folder, im)
        
        cv2.imwrite(dest_img_path, magnitude)

        print('Saved: ', dest_img_path)
