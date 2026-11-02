🩻 Explainable AI-Based Abnormality Detection Using Chest X-ray Images

An Explainable Artificial Intelligence (XAI) project that uses deep learning to identify abnormalities in chest X-ray images and visualize the regions influencing model predictions.

📌 Project Overview

Medical image classification using deep learning can provide useful predictions, but understanding why a model makes a particular prediction is equally important.

This project combines Convolutional Neural Networks (CNNs) and Explainable AI techniques to classify chest X-ray images and provide visual explanations of model decisions.

The system focuses on distinguishing between Normal and Pneumonia chest X-ray images.

🎯 Objectives

- Develop a deep learning-based chest X-ray classification system.
- Detect pneumonia-related abnormalities in X-ray images.
- Explore different pretrained CNN architectures.
- Apply Explainable AI techniques to interpret model predictions.
- Generate visual heatmaps to highlight important image regions.
- Build a user-friendly web application for image prediction.

✨ Key Features

- Chest X-ray image classification.
- Normal and Pneumonia category prediction.
- Multiple deep learning architectures.
- Grad-CAM-based visual explanations.
- Heatmap visualization of important image regions.
- Flask-based web application.
- Image upload and prediction interface.

🧠 Deep Learning Models

The project explores the following architectures:

- DenseNet
- EfficientNet
- InceptionResNet
- MobileNet
- Xception

🔍 Explainable Artificial Intelligence (XAI)

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to help interpret deep learning predictions.

Grad-CAM generates a heatmap that highlights image regions that contribute to a model's prediction.

This helps users understand which areas influenced the classification result.

🛠️ Technologies Used

Technology| Purpose
Python| Main programming language
TensorFlow / Keras| Deep learning
Flask| Web application
OpenCV| Image processing
NumPy| Numerical operations
Matplotlib| Visualization
Grad-CAM| Model explainability
HTML / CSS| Frontend interface

📂 Project Structure

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

⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/Xhkbasharat/explainable-xai-chest-xray.git

2. Navigate to the project directory

cd explainable-xai-chest-xray

3. Create a virtual environment

python -m venv venv

4. Activate the environment

Windows:

venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

6. Configure environment variables

Create a local ".env" file:

API_KEY=your_api_key
FLASK_SECRET_KEY=your_private_secret

Never upload ".env" or expose API keys in source code.

7. Run the application

python app.py

Open your browser at:

http://127.0.0.1:5000

🚀 Future Improvements

- Improve model accuracy and generalization.
- Add additional chest X-ray abnormality categories.
- Integrate more explainability techniques such as LIME.
- Improve heatmap visualization.
- Add model performance evaluation metrics.
- Develop a more comprehensive web interface.

⚠️ Disclaimer

This project is intended strictly for educational and research purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

👨‍💻 Author

Xhkbasharat

GitHub: "@Xhkbasharat" (https://github.com/Xhkbasharat)

⭐ Support

If you find this project useful, consider giving the repository a star!
