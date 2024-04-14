# CNNs and Their Fake Classification 

- Trained VGG16 models on benchmark datasets such as Imagenette, Coil, Caltech, Yale Faces, and others. 
- Examined fake classification by cropping out 20x20 background regions without objects in the image.
-	Utilized noise reduction techniques such as median filtering and transformations like Fourier and Wavelet to evaluate robustness against noise.
-	Creating an ensemble image classifier incorporating VGG16 models trained on original, Fourier-transformed, and Wavelet-transformed datasets
