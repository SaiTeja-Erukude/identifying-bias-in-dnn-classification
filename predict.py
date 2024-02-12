import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

test_data_dir = '../../data/imagenette2-320/test/'

# Get a list of folders in the directory
folders = []
for folder in os.listdir(test_data_dir):
    if os.path.isdir(os.path.join(test_data_dir, folder)):
        folders.append(folder)

num_of_classes = 10
total_preds = 200
correct_preds = 0

# Load the model
model = load_model("vgg16.h5")

for folder in folders:

    # Get the list of images in the folder
    images = []
    for im in os.listdir(os.path.join(test_data_dir, folder)):
        if im.endswith('.jpg') or im.endswith('.png') or im.endswith('.JPEG') or im.endswith('.jpeg'):
            images.append(im)

    for i in range(total_preds//num_of_classes):         
        # Choose an image
        img = images[i]

        # form the image path
        img_path = os.path.join(test_data_dir, folder, img)

        img = image.load_img(img_path, target_size=(224,224))
        img = np.asarray(img)

        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)

        print('Original:', folder)
        print('Predicted: ', folders[np.argmax(preds)])
        print(preds, end='\n\n')

        if folder == folders[np.argmax(preds)]:
            correct_preds += 1

print('Correct Preds: ', correct_preds)
print('Accuracy: ', correct_preds/total_preds)