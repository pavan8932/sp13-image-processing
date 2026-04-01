# Project Title: Image Smoothing and Sharpening with Histogram

from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
HIST_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ---------------- FILTERS ----------------

def apply_mean_filter(image):
    return cv2.blur(image, (25, 25))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (25, 25), 0)

def apply_laplacian_filter(image):
    # ① Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ② Apply Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # ③ Convert to absolute values
    laplacian = np.uint8(np.absolute(laplacian))

    # ④ Save PURE EDGE image (IMPORTANT)
    laplacian_edge = laplacian.copy()

    # ⑤ Convert to 3-channel for sharpening
    laplacian_colored = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    # ⑥ Sharpen original image
    sharpened = cv2.addWeighted(image, 1.5, laplacian_colored, -0.5, 0)

    return sharpened, laplacian_edge

# ---------------- HISTOGRAM FUNCTION ----------------

def save_histogram(image, filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    path = os.path.join(HIST_FOLDER, filename)
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

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        image = cv2.imread(input_path)

        # Apply filters
        mean_img = apply_mean_filter(image)
        gaussian_img = apply_gaussian_filter(image)
        laplacian_img, laplacian_edge = apply_laplacian_filter(image)
        # Save images
        mean_path = os.path.join(OUTPUT_FOLDER, 'mean_' + file.filename)
        gaussian_path = os.path.join(OUTPUT_FOLDER, 'gaussian_' + file.filename)
        laplacian_path = os.path.join(OUTPUT_FOLDER, 'laplacian_' + file.filename)
        laplacian_edge_path = os.path.join(OUTPUT_FOLDER, 'laplacian_edge_' + file.filename)

        cv2.imwrite(mean_path, mean_img)
        cv2.imwrite(gaussian_path, gaussian_img)
        cv2.imwrite(laplacian_path, laplacian_img)
        cv2.imwrite(laplacian_edge_path, laplacian_edge)
        # Save histograms
        hist_original = save_histogram(image, 'hist_original.png')
        hist_mean = save_histogram(mean_img, 'hist_mean.png')
        hist_gaussian = save_histogram(gaussian_img, 'hist_gaussian.png')
        hist_laplacian = save_histogram(laplacian_img, 'hist_laplacian.png')

        return render_template('index.html',
                               input_image=input_path,
                               mean_image=mean_path,
                               gaussian_image=gaussian_path,
                               laplacian_image=laplacian_path,
                               laplacian_edge_image=laplacian_edge_path,
                               hist_original=hist_original,
                               hist_mean=hist_mean,
                               hist_gaussian=hist_gaussian,
                               hist_laplacian=hist_laplacian)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)