'''
    Subtracts the weights of model2 (cropped images) from model1 (full images) and saves a new model
    An attempt to quantify the noise
'''

from keras.models import Sequential
from keras.models import load_model

# Load custom-trained VGG models
model_1 = load_model('vgg16/models/imagenette/vgg16_imagenette_20epochs_acc61.h5')
model_2 = load_model('vgg16/models/imagenette/vgg16_imagenette_cropped_20epochs.h5')

differenced_model = Sequential()

# Iterate through the layers of the models and subtract the weights
for layer_1, layer_2 in zip(model_1.layers, model_2.layers):
    # Check if the layers have weights (i.e., are trainable)
    if layer_1.weights:
        # Subtract the weights of the corresponding layers element-wise
        new_weights = [w1 - w2 for w1, w2 in zip(layer_1.get_weights(), layer_2.get_weights())]

        # Set the differenced weights to the new model
        differenced_model.add(layer_1.__class__(**layer_1.get_config()))
        differenced_model.layers[-1].set_weights(new_weights)        
    else:
        # If the layer does not have weights, just add it to the new model
        differenced_model.add(layer_1.__class__(**layer_1.get_config()))

# Save the new model
differenced_model.save('vgg16/models/imagenette/differenced_vgg16_imagenette_20epochs.h5')
print('Saved the differeneced model!')