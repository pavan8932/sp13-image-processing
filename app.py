# ================================
# 📌 PROJECT: Image Processing App
# ================================

from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np

# 🔥 IMPORTANT: Fix for server (Render)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- APP SETUP ----------------

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- FILTER FUNCTIONS ----------------

# Mean Filter → Simple averaging (strong blur)
def apply_mean_filter(image):
    return cv2.blur(image, (9, 9))

# Gaussian Filter → Smooth with weight (better than mean)
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (9, 9), 2)

# Laplacian Filter → Edge detection + sharpening
def apply_laplacian_filter(image):

    # Handle grayscale or color image
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert to visible format
    laplacian = cv2.convertScaleAbs(laplacian, alpha=2)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian = np.uint8(laplacian)

    # Pure edge image
    laplacian_edge = laplacian.copy()

    # Convert to 3 channel for sharpening
    laplacian_color = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    # Sharpen original image
    sharpened = cv2.addWeighted(image, 1.5, laplacian_color, -0.5, 0)

    return sharpened, laplacian_edge

# ---------------- HISTOGRAM FUNCTION ----------------

def save_histogram(image, filename):

    # Handle grayscale or color image
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot histogram
    plt.figure()
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path)
    plt.close()

    return path

# ---------------- MAIN ROUTE ----------------

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        # 📷 CAMERA IMAGE HANDLING
        if request.form.get("camera_image"):
            import base64
            data = request.form.get("camera_image").split(",")[1]
            image_bytes = base64.b64decode(data)

            input_path = os.path.join(UPLOAD_FOLDER, "camera.jpg")
            with open(input_path, "wb") as f:
                f.write(image_bytes)

        else:
            # 📁 FILE UPLOAD
            file = request.files.get('image')

            if not file or file.filename == '':
                return redirect(request.url)

            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

        # Read image
        image = cv2.imread(input_path)

        if image is None:
            return "Error loading image"

        # Detect image type
        image_type = "Grayscale" if len(image.shape) == 2 else "Color"

        # Apply filters
        mean_img = apply_mean_filter(image)
        gaussian_img = apply_gaussian_filter(image)
        laplacian_img, laplacian_edge = apply_laplacian_filter(image)

        # Save processed images
        mean_path = os.path.join(OUTPUT_FOLDER, 'mean.jpg')
        gaussian_path = os.path.join(OUTPUT_FOLDER, 'gaussian.jpg')
        laplacian_path = os.path.join(OUTPUT_FOLDER, 'laplacian.jpg')
        edge_path = os.path.join(OUTPUT_FOLDER, 'edge.jpg')

        cv2.imwrite(mean_path, mean_img)
        cv2.imwrite(gaussian_path, gaussian_img)
        cv2.imwrite(laplacian_path, laplacian_img)
        cv2.imwrite(edge_path, laplacian_edge)

        # Generate histograms
        hist_original = save_histogram(image, 'hist_original.png')
        hist_mean = save_histogram(mean_img, 'hist_mean.png')
        hist_gaussian = save_histogram(gaussian_img, 'hist_gaussian.png')
        hist_laplacian = save_histogram(laplacian_img, 'hist_laplacian.png')
        hist_edge = save_histogram(laplacian_edge, 'hist_edge.png')

        return render_template('index.html',
            image_type=image_type,
            input_image='/' + input_path,
            mean_image='/' + mean_path,
            gaussian_image='/' + gaussian_path,
            laplacian_image='/' + laplacian_path,
            laplacian_edge_image='/' + edge_path,
            hist_original='/' + hist_original,
            hist_mean='/' + hist_mean,
            hist_gaussian='/' + hist_gaussian,
            hist_laplacian='/' + hist_laplacian,
            hist_edge='/' + hist_edge
        )

    return render_template('index.html')


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)