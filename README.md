Explainable AI for Pneumonia Detection Using Chest X-Ray Images

Project Overview

Explainable AI for Pneumonia Detection Using Chest X-Ray Images is an AI-powered medical image analysis system developed to detect pneumonia from chest X-ray images while also providing an explanation of the model's prediction.

The project combines Deep Learning, Computer Vision, Lung Segmentation, Ensemble Learning, and Explainable Artificial Intelligence (XAI) into a Flask-based web application.

Unlike a traditional classification system that only provides a result such as "Pneumonia" or "Normal", this system uses Grad-CAM to generate a visual heatmap highlighting the regions of the chest X-ray that contributed to the model's prediction.

The system also includes patient management, prediction history, dashboard analytics, PDF report generation, webcam capture, image quality validation, and an AI Challenge Mode for evaluating prediction robustness.



---

Problem Statement

Pneumonia is a respiratory infection that can affect one or both lungs. Chest X-ray imaging is commonly used as part of the assessment of pneumonia.

Manual examination of X-ray images requires trained medical professionals and can be time-consuming. Deep learning models can assist in analysing medical images, but many deep-learning systems operate as black boxes.

A classification model may produce:

Prediction: PNEUMONIA
Confidence: 96%

However, the user may still ask:

> Why did the model predict pneumonia?



This project addresses this problem by combining pneumonia classification with Explainable AI.

The system attempts to provide:

Pneumonia/Normal prediction

Confidence score

Lung segmentation

Visual explanation using Grad-CAM

Robustness evaluation

Patient history

Automated PDF reports



---

Project Objectives

The major objectives of the project are:

1. Develop an AI-based pneumonia detection system using chest X-ray images.


2. Apply lung segmentation before classification.


3. Use multiple deep-learning models for prediction.


4. Combine predictions using ensemble learning.


5. Implement Explainable AI using Grad-CAM.


6. Highlight important regions of the X-ray image.


7. Provide prediction confidence.


8. Evaluate prediction robustness using AI Challenge Mode.


9. Develop a Flask-based web application.


10. Maintain patient and prediction history.


11. Generate professional PDF reports.


12. Provide dashboard-based analytics.


13. Create a user-friendly interface for X-ray analysis.




---

Core Concept

The complete AI pipeline is:

Chest X-Ray
     ↓
Image Quality Validation
     ↓
Image Preprocessing
     ↓
Lung Segmentation
     ↓
Segmented Lung Image
     ↓
Multiple CNN Models
     ↓
Ensemble Prediction
     ↓
Normal / Pneumonia
     ↓
Confidence Calculation
     ↓
Grad-CAM Explainability
     ↓
Robustness Testing
     ↓
Result + Report


---

Key Features

1. Pneumonia Detection

The system classifies chest X-ray images into:

NORMAL

PNEUMONIA


The classification is performed using multiple deep-learning models.


---

2. Lung Segmentation

A U-Net-based segmentation model is used to identify the lung region.

The purpose is to reduce the influence of irrelevant areas surrounding the lungs.

Original X-Ray
      ↓
U-Net
      ↓
Lung Mask
      ↓
Lung Region
      ↓
Classification

The segmentation model is represented by:

lung_segmentation_unet.h5


---

 Deep Learning Models

The classification pipeline uses an ensemble of multiple deep-learning architectures.

The configured models include:

DenseNet121

DenseNet121 uses dense connections between layers, allowing features from earlier layers to be reused by later layers.

DenseNet201

DenseNet201 is a deeper DenseNet architecture capable of learning complex visual representations.

Xception

Xception uses depthwise separable convolutions to improve computational efficiency while maintaining strong feature extraction capabilities.

EfficientNetB3

EfficientNetB3 provides a balance between model complexity and performance through compound scaling.

MobileNetV2

MobileNetV2 is a lightweight architecture designed for efficient computation.


---

 Ensemble Learning

Instead of relying on only one model, the system combines predictions from five classification models.

Conceptually:

DenseNet121
      ↓
DenseNet201
      ↓
Xception
      ↓
EfficientNetB3
      ↓
MobileNetV2
      ↓
Average Predictions
      ↓
Final Prediction

The configured ensemble score is calculated as:

Final Score =
(P1 + P2 + P3 + P4 + P5) / 5

where:

P1 = DenseNet121 prediction
P2 = DenseNet201 prediction
P3 = Xception prediction
P4 = EfficientNetB3 prediction
P5 = MobileNetV2 prediction

A configured threshold of:

0.5

is used for binary classification.

Conceptually:

Score >= 0.5
      ↓
PNEUMONIA

Score < 0.5
      ↓
NORMAL


---

 Explainable AI — Grad-CAM

One of the most important components of the project is Explainable Artificial Intelligence.

What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique commonly used with convolutional neural networks.

It produces a heatmap showing which regions of an image contributed strongly to a particular prediction.

Instead of showing only:

PNEUMONIA
96%

the system can also display a heatmap:

Original X-Ray
       +
Grad-CAM Heatmap
       ↓
Highlighted Important Regions

This provides a visual explanation of the model's decision.


---

Grad-CAM Workflow

Input X-Ray
     ↓
CNN Model
     ↓
Feature Maps
     ↓
Target Class
     ↓
Gradient Calculation
     ↓
Important Feature Weights
     ↓
Grad-CAM
     ↓
Heatmap
     ↓
Overlay on X-Ray

The generated heatmap is superimposed on the X-ray to make the important regions visually understandable.


---

Lung-Restricted Grad-CAM

The project combines:

Lung Segmentation + Grad-CAM

The lung segmentation mask can be used to restrict the explainability visualization to the lung region.

This creates the following pipeline:

X-Ray
  ↓
U-Net
  ↓
Lung Mask
  ↓
CNN
  ↓
Grad-CAM
  ↓
Apply Lung Mask
  ↓
Final Explainable Heatmap

This helps focus the visualization on the relevant anatomical region.


---

Confidence Score

The system calculates a confidence percentage based on the ensemble prediction.

For a pneumonia prediction:

Confidence = Prediction × 100

For a normal prediction:

Confidence = (1 - Prediction) × 100

For example:

Prediction = 0.93

Confidence = 0.93 × 100
           = 93%

The result can therefore be displayed as:

Prediction: PNEUMONIA
Confidence: 93%

> Confidence should not be interpreted as a clinically validated probability unless the model has undergone proper probability calibration and external validation.




---

Severity Classification

The application also provides a confidence-based severity category.

The configured logic is:

Confidence >= 90%
        ↓
HIGH

Confidence >= 70%
        ↓
MEDIUM

Confidence < 70%
        ↓
LOW

Important:

This is an application-defined confidence category, not a clinical measurement of pneumonia severity.


---

AI Challenge Mode

The project includes an AI Challenge Mode to evaluate prediction consistency.

The purpose is to determine whether a small change in the input image causes the model to produce a completely different prediction.

The system creates controlled variations of the original image.

Examples include:

Brightness increase

Brightness decrease

Contrast adjustment

Small rotation

Blur

Noise



---

Robustness Score

Each modified image is passed through the prediction pipeline.

If the prediction remains the same as the original image, it is considered consistent.

The robustness score can be calculated as:

Robustness Score =
Consistent Predictions
---------------------- × 100
Total Challenge Images

For example:

Original Prediction = PNEUMONIA

10 challenge images
9 produce PNEUMONIA

Robustness Score = 9/10 × 100
                 = 90%

This provides an additional experimental measure of model stability.


---

Dataset

Kaggle Chest X-Ray Dataset

The project uses a chest X-ray dataset for pneumonia classification.

The referenced dataset contains two major classes:

NORMAL
PNEUMONIA

The dataset counts used in the project documentation are:

Class	Images

Normal	1,341
Pneumonia	3,875
Total	5,216


The exact counts should be verified against the specific dataset version used for the final experiments.


---

Shenzhen Lung Segmentation Dataset

A lung segmentation dataset is also used for lung-region segmentation.

The segmentation data provides:

Chest X-ray images

Lung masks

Annotation information


These masks support the training or evaluation of the U-Net segmentation component.


---

Image Preprocessing

Before an X-ray is passed to the classification models, it undergoes preprocessing.

Typical steps include:

1. Image loading


2. Format conversion


3. Resizing


4. Pixel normalization


5. Lung segmentation


6. Lung mask application


7. Model input preparation



The configured classification input size is:

224 × 224 pixels


---

System Architecture

The overall architecture is:

USER
                      │
                      ▼
             ┌─────────────────┐
             │ Flask Web App   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Upload / Webcam │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Image Validation│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Preprocessing   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ U-Net Segmentation│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Segmented Lung  │
             └────────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     DenseNet121  DenseNet201  Xception
          │           │           │
          └───────────┼───────────┘
                      │
             ┌────────┴────────┐
             │ EfficientNetB3  │
             │   MobileNetV2   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Ensemble Model  │
             └────────┬────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
       Prediction         Grad-CAM
              │                │
              └───────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Result Dashboard│
             └───────┬─────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       History    PDF Report  Robustness


---

Web Application

The complete AI pipeline is integrated into a Flask web application.

Main Modules

Login Module

Provides secure authentication for authorized users.

Registration Module

Allows users to create accounts.

Welcome Page

Introduces the system and its purpose.

AI Dashboard

Provides an overview of system activity.

X-Ray Upload Module

Allows users to upload chest X-ray images.

Webcam Module

Allows supported users to capture an image through a webcam.

Prediction Module

Runs the AI pipeline and generates the prediction.

Result Module

Displays:

Prediction

Confidence

Severity

Original X-ray

Grad-CAM heatmap

Robustness information


Patient History

Stores previous analyses and patient records.

Medical News

Displays configured medical news information.

Weather Module

Provides live weather information where configured.

AI Chatbot

Provides an AI-based assistance interface where configured.

Pneumonia Awareness

Provides educational information related to pneumonia.

Feedback

Allows users to submit system feedback.

Settings

Provides application configuration options.

Logout

Securely ends the current user session.


---

Patient Management

The system provides patient record management.

Patient information can include:

Patient ID
Name
Age
Gender
Contact Details
Upload Date
Prediction
Confidence
Severity
Prediction History
Generated Reports

Administrative functionality can include:

Add patient

Search patient

View patient

Edit patient

Delete patient

View prediction history



---

Dashboard Analytics

The dashboard provides important system statistics.

Possible dashboard indicators include:

Total Patients
Total Scans
Pneumonia Detections
Normal Detections
Recent Uploads
Model Usage
System Status

This provides administrators and authorized users with a centralized overview of the system.


---

Professional PDF Report

After prediction, the system can generate a professional PDF report.

The report can contain:

Patient Information

Patient ID
Name
Age
Gender
Contact
Date

AI Analysis

Prediction
Confidence
Severity
Robustness Score

Images

Original Chest X-Ray
Grad-CAM Heatmap

The generated report provides a record of the AI-assisted analysis.


---

Database

The application uses SQLite for data storage.

A patient history table can contain:

id
name
age
gender
prediction
confidence
severity
scan_date
image

Additional database tables can be used for:

User accounts

Patient records

Prediction history

Feedback

Reports

System settings



---

Security Features

The application can incorporate:

User authentication

Password hashing

Role-based access

Session management

File type validation

Restricted record management


For production deployment, additional measures should be implemented, including:

Strong secret management

HTTPS

CSRF protection

Secure file storage

Database access controls

Audit logging

Encryption

Rate limiting



---

Suggested Project Structure

Explainable-AI-Pneumonia-Detection/
│
├── app.py
├── requirements.txt
│
├── model.h5
├── densenet201_model.h5
├── xception_model.h5
├── efficientnetb3_model.h5
├── mobilenetv2_model.h5
├── lung_segmentation_unet.h5
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── result.html
│   ├── patient_history.html
│   └── report.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   ├── heatmaps/
│   ├── reports/
│   └── qrcodes/
│
├── uploads/
│
├── database/
│   └── patients.db
│
└── README.md


---

Installation

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/Explainable-AI-Pneumonia-Detection.git

cd Explainable-AI-Pneumonia-Detection


---

2. Create Virtual Environment

Windows

python -m venv venv

venv\Scripts\activate

Linux/macOS

python3 -m venv venv

source venv/bin/activate


---

3. Install Dependencies

pip install -r requirements.txt

Typical dependencies include:

Flask
TensorFlow
Keras
OpenCV
NumPy
Pillow
Matplotlib
SQLite

Additional packages may be required depending on the final implementation.


---

Run the Application

Run:

python app.py

Then open:

http://127.0.0.1:5000/

in your browser.


---

Example Prediction Workflow

Step 1 — Login

The user logs into the system.

Step 2 — Upload X-Ray

The user uploads a chest X-ray image.

Step 3 — Image Validation

The application validates the uploaded image.

Step 4 — Preprocessing

The image is resized and normalized.

Step 5 — Lung Segmentation

The U-Net model extracts the lung region.

Step 6 — Classification

The segmented image is passed through the five classification models.

Step 7 — Ensemble

The individual predictions are averaged.

Step 8 — Prediction

The system determines:

NORMAL

or:

PNEUMONIA

Step 9 — Confidence

The application calculates the confidence percentage.

Step 10 — Grad-CAM

The system generates an explainability heatmap.

Step 11 — Robustness Testing

AI Challenge Mode tests the prediction against controlled image variations.

Step 12 — Save Record

Patient and prediction information is stored.

Step 13 — Generate Report

A PDF report is generated containing the analysis information.


---

Evaluation Metrics

The project can evaluate classification performance using standard metrics such as:

Accuracy

Accuracy =
(TP + TN) / (TP + TN + FP + FN)

Precision

Precision =
TP / (TP + FP)

Recall

Recall =
TP / (TP + FN)

F1 Score

F1 =
2 × Precision × Recall
----------------------
Precision + Recall

Additional Evaluation

The project can also evaluate:

Confusion Matrix

ROC Curve

AUC

Grad-CAM visualizations

Robustness Score

Segmentation performance



