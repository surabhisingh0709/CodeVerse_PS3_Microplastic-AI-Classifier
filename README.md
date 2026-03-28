Microplastic Morphology & Ecological Risk Classifier
Project Submission: CodeVerse_PS3
1. Project Overview
The Microplastic AI Classifier is an automated end-to-end analytical pipeline designed for the detection, classification, and ecological risk assessment of microplastics. This system enables researchers to perform high-precision field-testing of aquatic samples, moving from raw imagery to actionable environmental data in real-time.

2. Datasets and Preprocessing
To ensure robust model performance across varying aquatic conditions, the following data pipeline was implemented:

Dataset Selection
We utilized a curated dataset comprising over 2,000 microscopic images of microplastic morphotypes. The classes include:

Fibre: Thread-like synthetic structures.

Fragment: Jagged, irregular plastic particles.

Film: Thin, flexible planar membranes.

Pellet: Smooth, spherical or cylindrical primary plastics.

Image Preprocessing
Noise Reduction: Applied Gaussian Blurring to eliminate sensor artifacts and thermal noise.

Segmentation: Utilized Otsu's Thresholding for automated background subtraction to isolate particles from the aquatic matrix.

Standardization: Input images are resized to 224x224 pixels and normalized to a floating-point range of [-1, 1] to ensure compatibility with the neural network's input layer.

3. Model and Performance Metrics
The core of the system is a high-speed inference engine optimized for environmental field-work.

Model Architecture
Type: Optimized TensorFlow Lite (TFLite) Convolutional Neural Network.

Backbone: Based on the MobileNetV2 architecture, utilizing depthwise separable convolutions for high efficiency on edge-computing devices.

Performance Benchmarks
Accuracy: The model achieved a 92.4% categorical accuracy on the hold-out test set.

Latency: Average inference speed is approximately 75ms per image, enabling real-time classification.

Reliability: High precision in identifying high-risk "Fibres" against complex backgrounds.

4. Key Features
The system integrates several high-value tools for environmental monitoring:

Morphological Identification
Automated classification of microplastics into their respective scientific categories to identify pollution sources.

Automated Geometry Measurement
Precision measurement of particle dimensions in Micrometers (μm) using OpenCV-based contour analysis and Pixels Per Micron (PPM) calibration.

Ecological Threat Index (ETI)
A mathematical risk assessment visualized through a centered gauge. The index is derived from morphology-specific toxicity weights and inverse particle-size scaling.

Scientific Reporting
Integrated one-click PDF export functionality that generates standardized laboratory reports, including metadata, classification results, and risk assessments for audit trails.
