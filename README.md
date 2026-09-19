# Medical Imaging Explorer 🔬

An interactive image processing and computer vision exploration tool built with **Python**, **Streamlit**, and **OpenCV**.

> [!IMPORTANT]
> **Educational & Learning Disclaimer**:
> This software is created exclusively for educational, research, and algorithmic demonstration purposes. It is **NOT** a medical device and is **NOT** intended for clinical diagnosis, patient evaluation, treatment planning, or any medical decision-making.

---

## 🌟 Key Features

- **Image Ingestion**: Upload custom DICOM exports, PNG, JPEG, or JPG images, or explore using the built-in sample imaging test pattern.
- **Enhancement & Preprocessing**:
  - Dynamic Brightness and Contrast adjustments via PIL.
  - Convolutional 2D Laplacian Sharpening via OpenCV custom kernels.
  - Grayscale conversion and adjustable Gaussian Blur filtering.
- **Edge Detection & Segmentation**:
  - Canny Edge Detection with gradient-based boundary localization.
  - **Automatic Otsu Thresholding** for bimodal histogram binarization.
  - **Manual Thresholding** with interactive slider controls.
- **Morphological & Object Analysis**:
  - Contour detection (cv2.findContours) on binarized segmentations.
  - Object-area filtering to eliminate noise and small artifacts.
  - Filtered object counting and area quantification.
  - White-pixel percentage calculation (tissue/feature density indicator).
- **Region of Interest (ROI) Analysis**:
  - Configurable ROI bounding box (size and spatial coordinates).
  - High-resolution cropped ROI preview.
  - Localized ROI statistical metrics and intensity distributions.
- **Image Statistics & Distribution**:
  - Comprehensive statistical cards: Mean, Standard Deviation, Minimum, and Maximum pixel values.
  - Global vs. ROI pixel-intensity histograms with Matplotlib.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.9+ installed on your system.
- Git.

### 2. Clone the Repository
`ash
git clone https://github.com/<your-username>/medical-imaging-explorer.git
cd medical-imaging-explorer
`

### 3. Create a Virtual Environment
`ash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
`

### 4. Install Dependencies
`ash
pip install -r requirements.txt
`

### 5. Launch the Application
`ash
streamlit run app.py
`
The application will automatically open in your default browser at http://localhost:8501.

---

## 🌐 Deploying to Streamlit Community Cloud

This project is pre-configured for seamless deployment to **Streamlit Community Cloud**:

1. Push this repository to your **GitHub** account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select your repository, branch (main), and set **Main file path** to:
   `	ext
   app.py
   `
5. Click **Deploy!**

---

## 📦 Tech Stack & Dependencies

- **Streamlit**: Web interface and interactive component state management.
- **OpenCV (opencv-python-headless)**: High-performance computer vision algorithms (Canny, Otsu, Gaussian Blur, 2D Filtering, Contours).
- **NumPy**: Matrix operations and statistical calculations.
- **Pillow (PIL)**: Image loading, enhancement, and bounding-box drawing.
- **Matplotlib**: Statistical intensity histogram generation.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
