from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np

# 🔥 IMPORTANT FIX FOR RENDER (NO DISPLAY ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- FILTERS ----------------

def apply_mean_filter(image):
    return cv2.blur(image, (7, 7))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (7, 7), 1.5)

def apply_laplacian_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    laplacian = cv2.convertScaleAbs(laplacian, alpha=2)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian = np.uint8(laplacian)

    laplacian_edge = laplacian.copy()

    laplacian_colored = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(image, 1.5, laplacian_colored, -0.5, 0)

    return sharpened, laplacian_edge

# ---------------- HISTOGRAM ----------------

def save_histogram(image, filename):
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
        file = request.files.get('image')

        if not file or file.filename == '':
            return redirect(request.url)

        # Save input image
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Read image
        image = cv2.imread(input_path)

        # 🔥 FIX: Prevent crash if image not loaded
        if image is None:
            return "Error: Could not read image"

        # Apply filters
        mean_img = apply_mean_filter(image)
        gaussian_img = apply_gaussian_filter(image)
        laplacian_img, laplacian_edge = apply_laplacian_filter(image)

        # Save outputs
        mean_path = os.path.join(OUTPUT_FOLDER, 'mean_' + file.filename)
        gaussian_path = os.path.join(OUTPUT_FOLDER, 'gaussian_' + file.filename)
        laplacian_path = os.path.join(OUTPUT_FOLDER, 'laplacian_' + file.filename)
        laplacian_edge_path = os.path.join(OUTPUT_FOLDER, 'laplacian_edge_' + file.filename)

        cv2.imwrite(mean_path, mean_img)
        cv2.imwrite(gaussian_path, gaussian_img)
        cv2.imwrite(laplacian_path, laplacian_img)
        cv2.imwrite(laplacian_edge_path, laplacian_edge)

        # Histograms
        hist_original = save_histogram(image, 'hist_original.png')
        hist_mean = save_histogram(mean_img, 'hist_mean.png')
        hist_gaussian = save_histogram(gaussian_img, 'hist_gaussian.png')
        hist_laplacian = save_histogram(laplacian_img, 'hist_laplacian.png')
        hist_laplacian_edge = save_histogram(laplacian_edge, 'hist_laplacian_edge.png')

        return render_template('index.html',
            input_image='/' + input_path,
            mean_image='/' + mean_path,
            gaussian_image='/' + gaussian_path,
            laplacian_image='/' + laplacian_path,
            laplacian_edge_image='/' + laplacian_edge_path,
            hist_original='/' + hist_original,
            hist_mean='/' + hist_mean,
            hist_gaussian='/' + hist_gaussian,
            hist_laplacian='/' + hist_laplacian,
            hist_laplacian_edge='/' + hist_laplacian_edge
        )

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)