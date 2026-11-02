Explainable AI-Based Abnormality Detection Using Chest X-ray Images

An Explainable Artificial Intelligence (XAI) project that uses deep learning to identify abnormalities in chest X-ray images and visualize the regions influencing model predictions.

Project Overview

Medical image classification using deep learning can provide useful predictions, but understanding why a model makes a particular prediction is equally important.

This project combines Convolutional Neural Networks (CNNs) and Explainable AI techniques to classify chest X-ray images and provide visual explanations of model decisions.

The system focuses on distinguishing between Normal and Pneumonia chest X-ray images.

Objectives

- Develop a deep learning-based chest X-ray classification system.
- Detect pneumonia-related abnormalities in X-ray images.
- Explore different pretrained CNN architectures.
- Apply Explainable AI techniques to interpret model predictions.
- Generate visual heatmaps to highlight important image regions.
- Build a user-friendly web application for image prediction.

Key Features

- Chest X-ray image classification.
- Normal and Pneumonia category prediction.
- Multiple deep learning architectures.
- Grad-CAM-based visual explanations.
- Heatmap visualization of important image regions.
- Flask-based web application.
- Image upload and prediction interface.

Deep Learning Models

The project explores the following architectures:

- DenseNet
- EfficientNet
- InceptionResNet
- MobileNet
- Xception

Explainable Artificial Intelligence (XAI)

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to help interpret deep learning predictions.

Grad-CAM generates a heatmap that highlights image regions that contribute to a model's prediction.

This helps users understand which areas influenced the classification result.

Technologies Used

Technology| Purpose
Python| Main programming language
TensorFlow / Keras| Deep learning
Flask| Web application
OpenCV| Image processing
NumPy| Numerical operations
Matplotlib| Visualization
Grad-CAM| Model explainability
HTML / CSS| Frontend interface

Project Structure

XAI_Project/
│
├── app.py
├── xai.py
├── start.bat
│
├── dataset/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── Shenzhen/
│   ├── ann/
│   ├── img/
│   └── mask/
│
├── templates/
├── static/
│
├── train_densenet.py
├── train_efficientnet.py
├── train_inceptionresnet.py
├── train_mobilenet.py
├── train_xception.py
│
├── requirements.txt
└── README.md

Note: Dataset files, trained model weights, private databases, and credentials are excluded from the public GitHub repository.




⭐ Support

If you find this project useful, consider giving the repository a star!
