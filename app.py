import tensorflow as tf
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from flask import Flask, request, session, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import base64
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sqlite3
import feedparser
import requests
from datetime import datetime

from tensorflow.keras.models import load_model

lung_model = load_model(
    "lung_segmentation_unet.h5",
    compile=False
)


def segment_lungs(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to the segmentation model input size
    img = cv2.resize(img, (256, 256))

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Add batch and channel dimensions
    img_input = np.expand_dims(img, axis=(0, -1))

    # Predict lung mask
    pred = lung_model.predict(img_input, verbose=0)[0]

    # Convert prediction to binary mask
    mask = (pred > 0.5).astype(np.uint8)

    return mask
# ==========================
# AI CHALLENGE MODE
# ==========================


def generate_challenge_images(img):

    variants = {}

    # Brightness Increase
    variants["Brightness +10%"] = cv2.convertScaleAbs(
        img,
        alpha=1.0,
        beta=25
    )

    # Brightness Decrease
    variants["Brightness -10%"] = cv2.convertScaleAbs(
        img,
        alpha=1.0,
        beta=-25
    )

    # Contrast Increase
    variants["Contrast +10%"] = cv2.convertScaleAbs(
        img,
        alpha=1.1,
        beta=0
    )

    # Contrast Decrease
    variants["Contrast -10%"] = cv2.convertScaleAbs(
        img,
        alpha=0.9,
        beta=0
    )

    # Small Rotation (+2°)
    h, w = img.shape[:2]

    M1 = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        2,
        1.0
    )

    variants["Rotate +2°"] = cv2.warpAffine(
        img,
        M1,
        (w, h)
    )

    # Small Rotation (-2°)
    M2 = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        -2,
        1.0
    )

    variants["Rotate -2°"] = cv2.warpAffine(
        img,
        M2,
        (w, h)
    )

    # Gaussian Blur
    variants["Blur"] = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # Gaussian Noise
    noise = np.random.normal(
        0,
        8,
        img.shape
    ).astype(np.int16)

    noisy = np.clip(
        img.astype(np.int16) + noise,
        0,
        255
    ).astype(np.uint8)

    variants["Noise"] = noisy

    return variants


# REPORT PDF


# ==========================
# APP CONFIG
# ==========================
app = Flask(__name__)

API_KEY = "your_old_api_key"
def init_history_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age TEXT,
        gender TEXT,
        prediction TEXT,
        confidence REAL,
        severity TEXT,
        scan_date TEXT,
        image TEXT
    )
    """)

    conn.commit()
    conn.close()


init_history_db()
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    HEATMAP_FOLDER,
    exist_ok=True
)

# ==========================
# LOAD MODEL
# ==========================


densenet121 = tf.keras.models.load_model("model.h5")
model = densenet121

densenet201 = tf.keras.models.load_model("densenet201_model.h5")
xception = tf.keras.models.load_model("xception_model.h5")
# inceptionresnet = tf.keras.models.load_model("inceptionresnetv2_model.h5")
efficientnetb3 = tf.keras.models.load_model("efficientnetb3_model.h5")
mobilenetv2 = tf.keras.models.load_model("mobilenetv2_model.h5")
# ==========================
# MODEL ACCURACY EVALUATION
# ==========================

def calculate_model_accuracy(test_folder):

    y_true = []
    y_pred = []

    class_folders = {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }

    for class_name, true_label in class_folders.items():

        folder_path = os.path.join(
            test_folder,
            class_name
        )

        if not os.path.exists(folder_path):
            print("Folder not found:", folder_path)
            continue

        print(f"Evaluating {class_name} images...")

        for filename in os.listdir(folder_path):

            filepath = os.path.join(
                folder_path,
                filename
            )

            img = cv2.imread(filepath)

            if img is None:
                continue

            img = cv2.resize(
                img,
                (224, 224)
            )

            img = img.astype(
                "float32"
            ) / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            p1 = densenet121.predict(
                img,
                verbose=0
            )[0][0]

            p2 = densenet201.predict(
                img,
                verbose=0
            )[0][0]

            p3 = xception.predict(
                img,
                verbose=0
            )[0][0]

            p4 = efficientnetb3.predict(
                img,
                verbose=0
            )[0][0]

            p5 = mobilenetv2.predict(
                img,
                verbose=0
            )[0][0]

            score = (
                p1 + p2 + p3 + p4 + p5
            ) / 5

            predicted_label = (
                1 if score >= 0.5 else 0
            )

            y_true.append(true_label)
            y_pred.append(predicted_label)

            if len(y_true) == 0:

                print("No test images found.")

                return None

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            print("\n==============================")
            print("     ENSEMBLE MODEL RESULTS")
            print("==============================")

            print(
                f"Accuracy: {accuracy * 100:.2f}%"
            )

            print("\nClassification Report:")

            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=[
                        "NORMAL",
                        "PNEUMONIA"
                    ],
                    zero_division=0
                )
            )

            print("\nConfusion Matrix:")

            print(
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                )
            )

            print(
                "\nTotal Images Evaluated:",
                len(y_true)
            )

            print(
                "Actual NORMAL:",
                y_true.count(0)
            )

            print(
                "Actual PNEUMONIA:",
                y_true.count(1)
            )

            return accuracy * 100
# ==========================
# PREDICT FROM IMAGE
# ==========================


def predict_label_from_image(img):

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    p1 = densenet121.predict(img, verbose=0)[0][0]
    p2 = densenet201.predict(img, verbose=0)[0][0]
    p3 = xception.predict(img, verbose=0)[0][0]
    p4 = efficientnetb3.predict(img, verbose=0)[0][0]
    p5 = mobilenetv2.predict(img, verbose=0)[0][0]

    score = (p1 + p2 + p3 + p4 + p5) / 5

    if score >= 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"
# ==========================
# LOGIN PAGE
# ==========================


@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username"
        )

        password = request.form.get(
            "password"
        )

        conn = sqlite3.connect(
            "patients.db"
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM users
            WHERE username = ?
            AND password = ?
            """,
            (
                username,
                password
            )
        )

        user = cursor.fetchone()

        conn.close()

        if user:

            session["user"] = username

            return redirect(
                url_for("welcome")
            )

    return render_template(
        "login.html"
    )


# ==========================
# LOGOUT
# ==========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ==========================
# HOME
# ==========================

@app.route("/")
def home():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    return redirect(
        url_for("dashboard")
    )


# ==========================
# DASHBOARD
# ==========================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html"
    )


@app.route("/welcome")
def welcome():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "welcome.html"
    )

# ==========================
# STRICT XRAY VALIDATION
# ==========================


def is_chest_xray(filepath):

    try:

        img = cv2.imread(
            filepath
        )

        if img is None:
            return False

        img = cv2.resize(

            img,

            (300, 300)

        )

        gray_pixels = 0
        dark_pixels = 0

        total_pixels = (

            img.shape[0]

            *

            img.shape[1]

        )

        for row in img:

            for pixel in row:

                b, g, r = pixel

                brightness = (

                    int(r)

                    +

                    int(g)

                    +

                    int(b)

                ) / 3

                # XRAY grayscale
                if (

                    abs(
                        int(r)
                        -
                        int(g)
                    ) < 20

                    and

                    abs(
                        int(g)
                        -
                        int(b)
                    ) < 20
                ):

                    gray_pixels += 1

                # XRAY darker regions
                if (
                    brightness < 170
                ):

                    dark_pixels += 1

        gray_ratio = (

            gray_pixels

            /

            total_pixels

        )

        dark_ratio = (

            dark_pixels

            /

            total_pixels

        )

        return (
            gray_ratio > 0.90
            and
            dark_ratio > 0.60
        )

    except Exception as e:

        print(
            "XRAY VALIDATION ERROR:",
            e
        )

        return False

# ==========================
# GRAD CAM
# ==========================


def generate_gradcam(img_path):
    print("GRADCAM FUNCTION RUNNING")
    img = cv2.imread(img_path)

    original_img = img.copy()   # save original image

    img = cv2.resize(
        img,
        (224, 224)
    )
    img = cv2.resize(img, (224, 224))

    # Generate lung segmentation mask
    lung_mask = segment_lungs(img_path)

    # Resize mask to match image
    lung_mask = cv2.resize(lung_mask.astype(np.uint8), (224, 224))

    # Keep the original image for display
    display_img = img.copy()

    # Use the masked image only for prediction
    segmented_img = img * np.expand_dims(lung_mask, axis=-1)

    # Feed the segmented image to the model
    img = segmented_img

    img_array = np.expand_dims(
        img / 255.0,
        axis=0
    )

    p1 = densenet121.predict(img_array)[0][0]
    p2 = densenet201.predict(img_array)[0][0]
    p3 = xception.predict(img_array)[0][0]
    # p4 = inceptionresnet.predict(img_array)[0][0]
    p5 = efficientnetb3.predict(img_array)[0][0]
    p6 = mobilenetv2.predict(img_array)[0][0]

    prediction = float(p1 + p2 + p3 + p5 + p6) / 5

    print("RAW MODEL OUTPUT =", prediction)

    # FIX CONFIDENCE
    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction * 100
    else:
        label = "NORMAL"
        confidence = (1 - prediction) * 100

    confidence = round(confidence, 2)

    # ==========================
    # NORMAL -> ORIGINAL XRAY
    # ==========================
    if label == "NORMAL":
        infection_percentage = 0
        affected_region = "None"
        filename = os.path.basename(
            img_path
        )

        heatmap_filename = (
            "heat_" + filename
        )

        heatmap_path = os.path.join(
            HEATMAP_FOLDER,
            heatmap_filename
        )

        cv2.imwrite(
            heatmap_path,
            original_img
        )
        print("FINAL CONFIDENCE =", confidence)
        return (
            label,
            confidence,
            heatmap_filename,
            infection_percentage,
            affected_region
        )

    # ==========================
    # YOUR ORIGINAL GRADCAM
    # ==========================
    last_conv_layer = None

    for layer in reversed(
        model.layers
    ):

        if (
            len(layer.output.shape)
            == 4
        ):

            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # FIX ERROR
    if isinstance(
        heatmap,
        tf.Tensor
    ):

        heatmap = heatmap.numpy()

    heatmap = np.maximum(
        heatmap,
        0
    )

    # Restrict Grad-CAM to lung area
    lung_mask = cv2.resize(
        lung_mask.astype(np.float32),
        (heatmap.shape[1], heatmap.shape[0])
    )
    heatmap = heatmap * lung_mask

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

        # ==========================
        # INFECTION ANALYSIS
        # ==========================

        binary_mask = heatmap > 0.5

        infection_percentage = round(
            (
                np.sum(binary_mask)
                /
                binary_mask.size
            ) * 100,
            2
        )

        height, width = binary_mask.shape

        left_pixels = np.sum(
            binary_mask[:, :width//2]
        )

        right_pixels = np.sum(
            binary_mask[:, width//2:]
        )

        if left_pixels > right_pixels * 1.2:

            affected_region = "Left Lung"

        elif right_pixels > left_pixels * 1.2:

            affected_region = "Right Lung"

        else:

            affected_region = "Both Lungs"

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(

            heatmap,

            cv2.COLORMAP_JET

        )

        superimposed = cv2.addWeighted(
            display_img,
            0.6,
            heatmap,
            0.4,
            0
        )

        filename = os.path.basename(
            img_path
        )

        heatmap_filename = (
            "heat_"
            + filename
        )

        heatmap_path = os.path.join(

            HEATMAP_FOLDER,

            heatmap_filename

        )

        cv2.imwrite(

            heatmap_path,

            superimposed

        )

        return (
            label,
            confidence,
            heatmap_filename,
            infection_percentage,
            affected_region
        )

# ==========================
# PREDICT (defined above)
# ==========================


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")
    if not file:
        return render_template(
            "index.html",
            error="No file uploaded."
        )

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(

        UPLOAD_FOLDER,

        filename

    )

    file.save(filepath)

    # XRAY VALIDATION
    if not is_chest_xray(
        filepath
    ):

        os.remove(filepath)

        return render_template(

            "index.html",

            error="Only Chest X-Ray images are accepted."

        )

    name = request.form.get(
        "name"
    )

    age = request.form.get(
        "age"
    )

    gender = request.form.get(
        "gender"
    )

    (
        label,
        confidence,
        heatmap_file,
        infection_percentage,
        affected_region
    ) = generate_gradcam(
        filepath
    )
# AI CHALLENGE MODE
    original_img = cv2.imread(filepath)

    challenge_images = generate_challenge_images(original_img)

    same_predictions = 0

    for _, test_img in challenge_images.items():
        test_prediction = predict_label_from_image(test_img)
        if test_prediction == label:
            same_predictions += 1

    robustness_score = round(
        (same_predictions / len(challenge_images)) * 100,
        2
    )

    severity = "Normal"

    if label == "PNEUMONIA":
        if confidence >= 90:
            severity = "High"
        elif confidence >= 70:
            severity = "Medium"
        else:
            severity = "Low"

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    print("Saving patient:", name, age, gender,
          label, confidence, severity, filename)
    cursor.execute("""
    INSERT INTO patient_history
    (name, age, gender, prediction, confidence, severity, scan_date, image)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        age,
        gender,
        label,
        round(float(confidence), 2),
        severity,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename
    ))

    conn.commit()
    print("✅ Patient history inserted successfully")
    conn.close()

    session.update({
        "name": name,
        "age": age,
        "gender": gender,
        "result": label,
        "filename": filename,
        "confidence": float(round(float(confidence), 2)),
        "severity": severity,
        "infection_percentage": infection_percentage,
        "affected_region": affected_region,
        "robustness_score": robustness_score
    })

    return redirect(url_for("result"))


# ==========================
# RESULT PAGE
# ==========================

@app.route("/result")
def result():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "result.html",
        robustness_score=session.get("robustness_score"),
        prediction=session.get("result"),
        confidence=session.get("confidence"),
        severity=session.get("severity"),
        infection_percentage=session.get("infection_percentage"),
        affected_region=session.get("affected_region"),
        name=session.get("name"),
        age=session.get("age"),
        gender=session.get("gender"),
        image_path=url_for(
            "static",
            filename="uploads/" + session.get("filename")
        ),
        gradcam_path=url_for(
            "static",
            filename="heatmaps/heat_" + session.get("filename")
        )
    )


@app.route("/patient-history")
def patient_history():

    conn = sqlite3.connect("patients.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM patient_history
        ORDER BY id DESC
    """)

    history = cursor.fetchall()
    conn.close()

    return render_template(
        "patient_history.html",
        history=history
    )


@app.route("/system-status")
def system_status():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "system_status.html"
    )


@app.route("/settings")
def settings():

    return render_template(
        "settings.html"
    )


@app.route("/feedback")
def feedback():

    return render_template(
        "feedback.html"
    )


@app.route("/guest-login")
def guest_login():

    session["user"] = "guest"

    return redirect(
        url_for("dashboard")
    )


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(
            "patients.db"
        )

        cursor = conn.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO users(
                    username,
                    email,
                    password
                )
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    password
                )
            )

            conn.commit()
            conn.close()

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            conn.close()

            return "Username already exists. Try another username"

    return render_template(
        "register.html"
    )
# ==========================
# WEBCAM PAGE
# ==========================


@app.route("/webcam")
def webcam():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "webcam.html"
    )


# ==========================
# WEBCAM PREDICT
# ==========================

@app.route(
    "/webcam_predict",
    methods=["POST"]
)
def webcam_predict():

    if "user" not in session:

        return jsonify({

            "error":
            "Unauthorized"

        })

    try:

        data = request.json.get(
            "image"
        )

        if not data:

            return jsonify({

                "error":
                "No image received"

            })

        image_data = (

            base64.b64decode(
                data.split(",")[1]
            )

        )

        filename = (

            "webcam_"

            +

            datetime.now()
            .strftime(
                "%Y%m%d%H%M%S"
            )

            +

            ".png"

        )

        path = os.path.join(

            UPLOAD_FOLDER,

            filename

        )

        with open(
            path,
            "wb"
        ) as f:

            f.write(
                image_data
            )

        # XRAY VALIDATION
        if not is_chest_xray(
            path
        ):

            os.remove(path)

            return jsonify({

                "error":
                "Only Chest X-Ray images are accepted."

            })

        label, confidence, heatmap_file, infection_percentage, affected_region = generate_gradcam(
            path)

        # ==========================
        # AI CHALLENGE MODE
        # ==========================

        original_img = cv2.imread(path)

        challenge_images = generate_challenge_images(original_img)

        same_predictions = 0
        challenge_results = []

        for test_name, test_img in challenge_images.items():

            test_prediction = predict_label_from_image(test_img)

            challenge_results.append({
                "test": test_name,
                "prediction": test_prediction
            })

            if test_prediction == label:
                same_predictions += 1

        robustness_score = round(
            (same_predictions / len(challenge_images)) * 100,
            2
        )

        # SAVE SESSION
        session.update({
            "filename": filename,
            "result": label,
            "confidence": float(round(float(confidence), 2)),
            "name": "Webcam Patient",
            "age": "N/A",
            "gender": "N/A"
        })

        return jsonify({
            "success": True,
            "redirect": "/result"
        })

    except Exception as e:

        return jsonify({

            "error":
            str(e)

        })


# ==========================
# DOWNLOAD REPORT
# ==========================

@app.route("/download_report")
def download_report():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    pdf_path = "report.pdf"

    doc = SimpleDocTemplate(
        pdf_path
    )

    styles = (
        getSampleStyleSheet()
    )

    content = []

    now = datetime.now()

    report_id = (

        "XAI-"

        +

        now.strftime(
            "%Y%m%d%H%M%S"
        )

    )

    # =====================
    # HEADER
    # =====================

    header = Table([

        [
            "AI MEDICAL HOSPITAL",
            "AI Diagnosis Report"
        ]

    ],
        colWidths=[300, 200])

    header.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor(
                    "#0a2540"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.white
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                16
            )

        ])

    )

    content.append(
        header
    )

    content.append(
        Spacer(1, 20)
    )

    # =====================
    # PATIENT INFO
    # =====================

    content.append(

        Paragraph(

            "<b>Patient Information</b>",

            styles["Heading2"]

        )

    )

    patient_table = Table([

        [
            "Name",
            session.get(
                "name",
                "N/A"
            )
        ],

        [
            "Age",
            session.get(
                "age",
                "N/A"
            )
        ],

        [
            "Gender",
            session.get(
                "gender",
                "N/A"
            )
        ]

    ])

    patient_table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            )

        ])

    )

    content.append(
        patient_table
    )

    content.append(
        Spacer(1, 20)
    )

    # =====================
    # RESULT
    # =====================

    content.append(

        Paragraph(

            "<b>Diagnosis Result</b>",

            styles["Heading2"]

        )

    )

    content.append(

        Paragraph(

            f"Prediction: {session.get('result','N/A')}",

            styles["Normal"]

        )

    )

    content.append(

        Paragraph(

            f"Confidence: {session.get('confidence','N/A')}%",

            styles["Normal"]

        )

    )

    content.append(
        Spacer(1, 20)
    )

    # =====================
    # XRAY + GRADCAM
    # =====================

    filename = session.get(
        "filename",
        ""
    )

    original_path = os.path.join(

        UPLOAD_FOLDER,

        filename

    )

    heatmap_path = os.path.join(

        HEATMAP_FOLDER,

        "heat_"
        + filename

    )

    if (

        os.path.exists(
            original_path
        )

        and

        os.path.exists(
            heatmap_path
        )

    ):

        image_table = Table([

            [
                "Chest X-Ray",
                "GradCAM"
            ],

            [

                Image(
                    original_path,
                    width=2.5*inch,
                    height=2.5*inch
                ),

                Image(
                    heatmap_path,
                    width=2.5*inch,
                    height=2.5*inch
                )

            ]

        ])

        content.append(
            image_table
        )

    content.append(
        Spacer(1, 30)
    )

    content.append(

        Paragraph(

            "Doctor Signature: __________________",

            styles["Normal"]

        )

    )

    doc.build(
        content
    )

    return send_file(

        pdf_path,

        as_attachment=True

    )

# ==========================
# IFRAME ROUTES FOR INDEX
# ==========================


@app.route("/dashboard-page")
def dashboard_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html"
    )


@app.route("/upload")
def upload_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "upload.html"
    )


@app.route("/history")
def history_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "history.html"
    )


@app.route("/status")
def status_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "system_status.html"
    )


@app.route("/chatbot")
def chatbot_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "chatbot.html"
    )


@app.route("/feedback")
def feedback_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "feedback.html"
    )

@app.route("/medical-news")
def medical_news():

    feed = feedparser.parse(
        "https://news.google.com/rss/search?q=pneumonia+OR+medical+AI"
    )

    articles = []

    for article in feed.entries[:10]:
        articles.append({
            "title": article.title,
            "link": article.link,
            "published": getattr(article, "published", "Unknown Date")
        })

    return render_template(
        "medical_news.html",
        articles=articles
    )

@app.route("/weather")
def weather():

    city = "Srinagar"

    # Get weather data
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    weather = requests.get(weather_url).json()

    # Get coordinates for AQI
    lat = weather["coord"]["lat"]
    lon = weather["coord"]["lon"]

    # Get Air Quality data
    aqi_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )

    aqi = requests.get(aqi_url).json()

    return render_template(
        "weather.html",
        weather=weather,
        aqi=aqi
    )

@app.route("/settings")
def settings_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "settings.html"
    )


@app.route("/awareness")
def awareness_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "awareness.html"
    )
# ==========================
# RUN APP
# ==========================


if __name__ == "__main__":

    accuracy = calculate_model_accuracy(
        "dataset"
    )

    if accuracy is not None:
        print(
            f"\nFINAL TEST ACCURACY: {accuracy:.2f}%"
        )

        app.run(
            debug=True
        )