# ---------------- IMPORTS ----------------
from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')  # FIX for server
import matplotlib.pyplot as plt

# ---------------- APP CONFIG ----------------
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- FILTERS ----------------

# Mean Filter
def apply_mean_filter(image):
    return cv2.blur(image, (25, 25))

# Gaussian Filter
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# 🔥 FINAL IMPROVED LAPLACIAN FILTER
def apply_laplacian_filter(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert to abs
    laplacian = cv2.convertScaleAbs(laplacian)

    # Enhance edges
    laplacian = cv2.convertScaleAbs(laplacian, alpha=4)

    # Threshold → pure edges
    _, laplacian_edge = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

    # Convert for sharpening
    laplacian_colored = cv2.cvtColor(laplacian_edge, cv2.COLOR_GRAY2BGR)

    # Sharpen image
    sharpened = cv2.addWeighted(image, 1.7, laplacian_colored, -0.7, 0)

    return sharpened, laplacian_edge

# ---------------- HISTOGRAM ----------------

def save_histogram(image, filename):

    # Handle grayscale safely
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.hist(gray.ravel(), bins=256)

    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path)
    plt.close()

    return path

# ---------------- ROUTE ----------------

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        # Save input image
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        image = cv2.imread(input_path)

        if image is None:
            return "Error loading image"

        # Detect type
        image_type = "Grayscale" if len(image.shape) == 2 else "Color"

        # Apply filters
        mean_img = apply_mean_filter(image)
        gaussian_img = apply_gaussian_filter(image)
        laplacian_img, laplacian_edge = apply_laplacian_filter(image)

        # Save outputs
        mean_path = os.path.join(OUTPUT_FOLDER, 'mean_' + file.filename)
        gaussian_path = os.path.join(OUTPUT_FOLDER, 'gaussian_' + file.filename)
        laplacian_path = os.path.join(OUTPUT_FOLDER, 'laplacian_' + file.filename)
        edge_path = os.path.join(OUTPUT_FOLDER, 'edge_' + file.filename)

        cv2.imwrite(mean_path, mean_img)
        cv2.imwrite(gaussian_path, gaussian_img)
        cv2.imwrite(laplacian_path, laplacian_img)
        cv2.imwrite(edge_path, laplacian_edge)

        # Histograms
        hist_original = save_histogram(image, 'hist_original.png')
        hist_mean = save_histogram(mean_img, 'hist_mean.png')
        hist_gaussian = save_histogram(gaussian_img, 'hist_gaussian.png')
        hist_laplacian = save_histogram(laplacian_img, 'hist_laplacian.png')
        hist_edge = save_histogram(laplacian_edge, 'hist_edge.png')

        return render_template('index.html',
            input_image='/' + input_path,
            mean_image='/' + mean_path,
            gaussian_image='/' + gaussian_path,
            laplacian_image='/' + laplacian_path,
            laplacian_edge_image='/' + edge_path,
            hist_original='/' + hist_original,
            hist_mean='/' + hist_mean,
            hist_gaussian='/' + hist_gaussian,
            hist_laplacian='/' + hist_laplacian,
            hist_edge='/' + hist_edge,
            image_type=image_type
        )

    return render_template('index.html')


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)