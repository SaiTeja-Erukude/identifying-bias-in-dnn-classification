'''
Ensemble Classifier Using Mode
- Predicts the class of an image using 3 models
- Considers the mode of the output of all the models
'''

import os
import cv2
import pywt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from statistics import mode


def fourierTransform(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  

    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)

    # calculate the magnitude of the Fourier Transform
    magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0], fourier_shift[:,:,1]))
    
    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    cv2.imwrite('FourierTest.JPEG', magnitude)
    img = image.load_img('FourierTest.JPEG', target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    
    return img


def waveletTransform(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
              
    # Perform 2D Discrete Wavelet Transform (DWT)
    coeffs = pywt.dwt2(img, 'haar')

    # Extract approximation, horizontal, vertical, and diagonal coefficients
    cA, (cH, cV, cD) = coeffs

    cv2.imwrite('WaveletTest.JPEG', cA.astype(np.uint8))

    img = image.load_img('WaveletTest.JPEG', target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    
    return img


def ensemblePredict(img_path): 

    # preprocess the image
    img = image.load_img(img_path, target_size=(224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    # apply Fourier transform to the image
    fourier_img = fourierTransform(img_path)

    # apply Wavelet transform to the image
    wavelet_img = waveletTransform(img_path)

    pred_list = []

    # predict using all the models
    base_preds = base_model.predict(img)
    pred_list.append(folders[np.argmax(base_preds)])
    
    fourier_preds = fourier_model.predict(fourier_img)
    pred_list.append(folders[np.argmax(fourier_preds)])
    
    wavelet_preds = wavelet_model.predict(wavelet_img)
    pred_list.append(folders[np.argmax(wavelet_preds)])

    # consider the mode of all predictions
    pred = mode(pred_list)

    return pred, pred_list
    
    
if __name__ == '__main__':

    test_data_dir = '../data/imagenette/test/'

    # Get a list of folders in the directory
    folders = []
    for folder in os.listdir(test_data_dir):
        if os.path.isdir(os.path.join(test_data_dir, folder)):
            folders.append(folder)

    total_preds = 0
    correct_preds = 0

    # Load the models
    base_model = load_model("./vgg16/models/vgg16_rms_acc60.h5")              # base model trained on original data
    fourier_model = load_model("./vgg16/models/vgg16_fourier_acc43.h5")       # model trained on fourier transformed data
    wavelet_model = load_model("./vgg16/models/vgg16_wavelet_acc43.h5")       # model trained on wavelet transformed data

    print('All models loaded!\n\n')

    for folder in folders:

        # Get the list of images in the folder
        images = []
        for im in os.listdir(os.path.join(test_data_dir, folder)):
            if im.endswith('.jpg') or im.endswith('.png') or im.endswith('.JPEG') or im.endswith('.jpeg'):
                images.append(im)

        for img in images:
            # form the image path
            img_path = os.path.join(test_data_dir, folder, img)

            # ensemble predict using all the models
            pred, pred_list = ensemblePredict(img_path)

            print('Original:', folder)
            print('Predicted: ', pred)
            print(pred_list, end='\n\n')

            total_preds += 1
            if folder == pred:
                correct_preds += 1

    print('Correct Preds: ', correct_preds)
    print('Total Preds: ', total_preds)
    print('Accuracy: ', correct_preds/total_preds)

    # delete transformed test images
    if os.path.exists('WaveletTest.JPEG'):
        os.remove('WaveletTest.JPEG')
    if os.path.exists('FourierTest.JPEG'):
        os.remove('FourierTest.JPEG')