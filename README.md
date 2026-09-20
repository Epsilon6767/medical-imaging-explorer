# Medical Imaging Explorer

An image processing and analysis tool built with Python, OpenCV, and Streamlit.

---

## Features

- **Pipeline Stages Preview**: Reference examples of Original X-ray, Canny Edges, Thresholded Image, and Contours.
- **Image Loading**: Supports PNG, JPG, and JPEG image uploads, plus an option to use the reference X-ray.
- **Image Adjustments**:
  - Brightness and contrast adjustments via PIL.
  - Sharpening using an OpenCV 3x3 Laplacian convolution kernel.
- **Preprocessing and Edge Detection**:
  - Grayscale conversion.
  - Gaussian blur with adjustable odd kernel size.
  - Canny edge detection.
- **Segmentation**:
  - Manual thresholding.
  - Automatic Otsu thresholding.
- **Object Analysis**:
  - External contour detection using OpenCV.
  - Minimum object area filtering.
  - Detected and filtered object counts.
  - White pixel percentage calculation.
- **Region of Interest (ROI)**:
  - Adjustable bounding box (size and coordinates).
  - Cropped ROI view.
  - Localized pixel intensity statistics.
- **Statistics and Histograms**:
  - Mean, standard deviation, min, and max intensity for both the full image and the selected ROI.
  - Pixel intensity histograms (256 bins) plotted with Matplotlib.

---

## Local Setup

### 1. Prerequisites
- Python 3.9 or higher
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/epsilon6767/medical-imaging-explorer.git
cd medical-imaging-explorer
```

### 3. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

---

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub:
   ```bash
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**.
4. Set:
   - **Repository**: epsilon6767/medical-imaging-explorer
   - **Branch**: main
   - **Main file path**: app.py
5. Click **Deploy!**

---

## Dependencies

- streamlit
- opencv-python-headless
- numpy
- pillow
- matplotlib
- pandas