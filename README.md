🫁 Explainable AI for Pneumonia Detection

Deep Learning • Computer Vision • Ensemble Learning • Explainable AI

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=900&color=00E5FF&center=true&vCenter=true&width=850&lines=AI-Powered+Pneumonia+Detection;Explainable+Deep+Learning;Grad-CAM+Visual+Interpretability;Computer+Vision+for+Medical+Imaging" alt="Animated project tagline" />
</p><p align="center">
  <img src="https://img.shields.io/badge/Domain-Medical%20AI-00E5FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20Ensemble-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-00C853?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Flask-orange?style=for-the-badge&logo=flask&logoColor=white" />
</p><p align="center">
  <b>Making medical image classification more interpretable through Explainable AI.</b>
</p>---

🌌 Project Overview

Explainable AI for Pneumonia Detection Using Chest X-Ray Images is an AI-powered medical image analysis system designed to classify chest X-ray images into two categories:

- 🟢 NORMAL
- 🔴 PNEUMONIA

The project combines Deep Learning, Computer Vision, Lung Segmentation, Ensemble Learning, and Explainable AI (XAI) in a Flask-based web application.

Rather than presenting only a model prediction, the system is designed to provide visual explanations using Grad-CAM heatmaps, helping users understand which image regions influenced the model's output.

It also incorporates patient management, prediction history, dashboard analytics, PDF report generation, image quality validation, webcam capture, and an AI Challenge Mode for testing prediction consistency under image transformations.

🎯 Key Objectives

- Develop a deep-learning system for pneumonia classification.
- Segment lung regions using a U-Net model.
- Integrate multiple CNN architectures.
- Apply ensemble learning to combine model predictions.
- Generate Grad-CAM visual explanations.
- Highlight important regions in chest X-ray images.
- Display prediction confidence scores.
- Evaluate model robustness under image transformations.
- Build an interactive Flask-based application.
- Maintain patient records and prediction history.
- Generate downloadable PDF reports.

---

🧠 System Architecture

The project follows a multi-stage medical image analysis pipeline.

              ┌──────────────────────┐
              │  Chest X-Ray Upload  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ Image Quality Check  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   Preprocessing      │
              │ Resize & Normalize   │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  U-Net Segmentation  │
              │    Lung Extraction   │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  CNN Model Ensemble  │
              └──────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   DenseNet121      DenseNet201       Xception
        │                │                │
   EfficientNetB3    MobileNetV2          │
        └────────────────┼────────────────┘
                         │
              ┌──────────▼───────────┐
              │ Ensemble Prediction  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ NORMAL / PNEUMONIA   │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ Grad-CAM Explanation │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ Results & PDF Report │
              └──────────────────────┘

---

⚙️ Core AI Components

1. 🫁 Pneumonia Classification

The classification task uses five configured convolutional neural network architectures:

Model| Architecture
DenseNet121| Dense Convolutional Network
DenseNet201| Deep Dense Convolutional Network
Xception| Depthwise Separable Convolutions
EfficientNetB3| Efficient Convolutional Network
MobileNetV2| Lightweight CNN Architecture

The models classify chest X-ray images into NORMAL or PNEUMONIA categories.

2. 🧩 Lung Segmentation — U-Net

A U-Net-based segmentation model is used to identify lung regions in chest X-ray images.

Model file: "lung_segmentation_unet.h5"

Key responsibilities:

- Generate lung segmentation masks.
- Identify relevant lung regions.
- Reduce the influence of surrounding image areas.
- Support lung-focused visual explanations.

3. 🔬 Ensemble Learning

The system combines predictions from five configured CNN models using average probability aggregation.

Ensemble formula:

[
P_{\text{ensemble}} =
\frac{P_1+P_2+P_3+P_4+P_5}{5}
]

Where each (P_i) represents the pneumonia score from one model.

Classification threshold: 0.5

- Score ≥ 0.5 → PNEUMONIA
- Score < 0.5 → NORMAL

4. 🔥 Explainable AI — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize image regions that contribute to a model's prediction.

Explainability workflow:

1. Generate a model prediction.
2. Calculate gradients with respect to a target convolutional layer.
3. Produce a class activation heatmap.
4. Overlay the heatmap on the original X-ray.
5. Apply a lung segmentation mask for lung-restricted visualization.

This helps make model outputs more interpretable, although heatmaps do not establish clinical reasoning or prove that a model is focusing on medically valid features.

5. 🧪 AI Challenge Mode

The AI Challenge Mode is designed to evaluate prediction consistency under controlled image transformations.

Supported transformations include:

- Brightness increase and decrease.
- Contrast adjustment.
- Small image rotations.
- Gaussian blur.
- Noise addition.

Robustness score:

[
R = \frac{\text{Consistent Predictions}}
{\text{Total Challenge Images}} \times 100
]

This is an application-defined consistency measure, not a substitute for formal clinical validation or comprehensive robustness evaluation.

---

📊 Dataset Information

Chest X-Ray Dataset

The project documentation describes the following dataset distribution:

Category| Images
NORMAL| 1,341
PNEUMONIA| 3,875
Total| 5,216

Lung Segmentation Dataset

Shenzhen Lung Segmentation Dataset

Used for lung image segmentation and mask-based processing.

Dataset counts and preprocessing details should be checked against the exact dataset versions used in the experiments.

---

🖥️ Flask Web Application

The project brings the AI pipeline into an interactive web application.

Application Modules

Module| Functionality
Authentication| User registration and login
AI Dashboard| Main application dashboard
X-Ray Upload| Upload chest X-ray images
Webcam Capture| Capture images through a webcam
Image Validation| Check image quality
Prediction| Run pneumonia classification
Explainability| Display Grad-CAM visualizations
Patient Management| Maintain patient information
Prediction History| Review previous predictions
PDF Reports| Generate prediction reports
Dashboard Analytics| Display application statistics
AI Chatbot| Interactive assistance
Pneumonia Awareness| Educational information
Feedback| Collect user feedback
Settings| Manage application preferences

---

📈 Evaluation & Performance Analysis

The project evaluation framework includes:

Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC

Explainability analysis

- Grad-CAM heatmap visualization.
- Lung-restricted Grad-CAM visualization.

Robustness analysis

- Prediction consistency under image transformations.
- AI Challenge Mode results.

Segmentation analysis

- Lung mask visualization.
- Segmentation quality assessment.

Actual metric values should be added only after running experiments and validating the results.

---

🛠️ Technology Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,flask,html,css,js,sqlite,git,github" alt="Technology stack" />
</p>Category| Technologies
Programming Language| Python
Deep Learning| TensorFlow, Keras
Computer Vision| OpenCV
AI Architectures| DenseNet, Xception, EfficientNet, MobileNetV2
Segmentation| U-Net
Explainable AI| Grad-CAM
Backend| Flask
Frontend| HTML, CSS, JavaScript, Bootstrap
Database| SQLite
Data Processing| NumPy, Pillow
Visualization| Matplotlib
Reporting| PDF generation tools

---

🚀 Installation & Setup

1. Clone the Repository

git clone https://github.com/Xhkbasharat/explainable-xai-chest-xray.git
cd explainable-xai-chest-xray

2. Create a Virtual Environment

python -m venv venv

Activate it:

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Configure Model Files

Ensure the required trained model files are available in the locations expected by the application, including:

lung_segmentation_unet.h5

The five classification model files must also be placed in their configured paths.

5. Run the Application

python app.py

The startup command, model filenames, and dependency list should be adjusted to match the actual repository implementation.

---

📁 Project Structure

An illustrative organization for the application:

explainable-xai-chest-xray/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── lung_segmentation_unet.h5
│   ├── densenet121/
│   ├── densenet201/
│   ├── xception/
│   ├── efficientnetb3/
│   └── mobilenetv2/
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── uploads/
│
├── templates/
│
├── database/
│
├── utils/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── ensemble.py
│   ├── gradcam.py
│   └── evaluation.py
│
├── reports/
│
└── README.md

Illustrative structure; actual filenames and directories may differ.

---

🌟 What Makes This Project Interesting?

- Multiple CNN architectures: Combines five configured deep-learning models.
- Lung-aware processing: Integrates U-Net segmentation into the analysis pipeline.
- Visual interpretability: Uses Grad-CAM to show influential image regions.
- Ensemble predictions: Aggregates model outputs through average probability.
- Robustness experimentation: Tests consistency under controlled image changes.
- Complete application workflow: Connects AI inference with patient records, history, and reporting.

---

⚠️ Medical Disclaimer

This project is intended for educational, research, and software-development purposes only.

It is not a clinically validated diagnostic device and must not be used as a substitute for professional medical evaluation.

Model predictions, confidence scores, Grad-CAM heatmaps, and application-defined severity categories are not clinical diagnoses or validated measures of disease severity. The system requires appropriate external validation before any clinical use.

---

👨‍💻 Author

Sheikh Basharat Akbar

AI Writer at Mercor | AI Trainer at micro1 | AI/ML & Software Engineer

MCA Graduate — Islamic University of Science & Technology

Areas of Interest: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Explainable AI, NLP, and intelligent software systems.

<p align="center">
  <a href="https://www.linkedin.com/in/sheikh-basharat-389980225/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="https://github.com/Xhkbasharat">
    <img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>---

⭐ Support

If you find this project interesting, consider giving the repository a ⭐.

Suggestions, feedback, and technical discussions are welcome.

<p align="center">
  <b>Explainable AI for Pneumonia Detection</b><br/>
  <i>Building AI systems that don't just predict — they help explain.</i>
</p><p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00E5FF,50:6A00F4,100:00C853&height=110&section=footer" width="100%" />
</p>
