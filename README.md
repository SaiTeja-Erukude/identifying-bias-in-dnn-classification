# Identifying bias in deep neural networks classification

### Problem Statement
Deep Neural Networks, particularly CNNs, function as "black boxes," making it difficult to ascertain which features influence classification outcomes. This opacity can introduce biases from background information that are hard to detect. Identifying and distinguishing dataset biases, contextual information, and background noise is essential for reliable performance evaluation.

### Proposed Methods
- Cropping Background Segments: Crop 20x20 blank background segments from images to analyze their impact on classification.
- Image Tiling: Divide images into smaller, non-overlapping tiles of varying sizes and shuffle them randomly. This disrupts object features, making classification more challenging.
- Transform Techniques: Utilize Fourier and Wavelet transforms, along with Median filters, to recover and analyze background noise. This method effectively distinguishes between contextual information and background noise, allowing for alerts on background noise presence without needing explicit background data.
