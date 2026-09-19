"""
Medical Imaging Explorer
========================
An interactive educational image processing tool built with Streamlit, OpenCV, and Pillow.

NOTE: This application is for educational and algorithmic exploration only.
It is NOT designed or approved for clinical or medical diagnostic use.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt
import os
import pandas as pd


# ------------------------------------------------------------------------------
# 1. Image Processing & Helper Functions
# ------------------------------------------------------------------------------

def load_image(uploaded_file, sample_path=None) -> Image.Image:
    """
    Loads an image from either an uploaded file or a sample file path,
    converting it to RGB to avoid alpha-channel/format issues.
    """
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
    elif sample_path and os.path.exists(sample_path):
        img = Image.open(sample_path)
    else:
        return None
    return img.convert("RGB")


def apply_enhancements(
    pil_image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpen_strength: float = 0.0
) -> Image.Image:
    """
    Applies brightness, contrast adjustments, and optional 2D kernel sharpening.
    - Brightness and Contrast use PIL ImageEnhance.
    - Sharpening uses an OpenCV 3x3 Laplacian-style convolution filter.
    """
    # Adjust brightness and contrast
    enhanced = ImageEnhance.Brightness(pil_image).enhance(brightness)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)

    # Apply sharpening if requested
    if sharpen_strength > 0:
        img_array = np.array(enhanced)
        center = 5.0 + sharpen_strength
        # Laplacian-based sharpening kernel with adjustable center weight
        kernel = np.array([
            [0, -1, 0],
            [-1, center, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        # Apply 2D convolution; cv2 automatically handles uint8 saturation
        sharpened_arr = cv2.filter2D(img_array, -1, kernel)
        enhanced = Image.fromarray(sharpened_arr)

    return enhanced


def to_grayscale_array(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL image to an 8-bit single-channel NumPy array."""
    return np.array(pil_image.convert("L"))


def apply_gaussian_blur(gray_array: np.ndarray, blur_strength: int) -> np.ndarray:
    """
    Applies Gaussian blur to smooth high-frequency noise.
    Kernel size must be an odd positive integer.
    """
    if blur_strength > 1:
        # Ensure odd kernel size (e.g., 3, 5, 7, ...)
        ksize = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
        return cv2.GaussianBlur(gray_array, (ksize, ksize), 0)
    return gray_array


def detect_edges_canny(
    gray_array: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> np.ndarray:
    """Computes Canny edge detection on a single-channel grayscale array."""
    return cv2.Canny(gray_array, low_threshold, high_threshold)


def apply_thresholding(
    gray_array: np.ndarray,
    use_otsu: bool = False,
    manual_threshold_val: int = 128
) -> tuple[int, np.ndarray]:
    """
    Segments the image into binary foreground and background.
    - When use_otsu is True: automatically computes the optimal threshold using
      Otsu's bimodal histogram method and returns cv2.threshold's binary image.
    - When use_otsu is False: applies the user-selected manual threshold.
    """
    if use_otsu:
        # Otsu's thresholding calculates the threshold automatically from image histogram
        computed_thresh, binary_mask = cv2.threshold(
            gray_array,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return int(computed_thresh), binary_mask
    else:
        # Manual thresholding
        _, binary_mask = cv2.threshold(
            gray_array,
            manual_threshold_val,
            255,
            cv2.THRESH_BINARY
        )
        return manual_threshold_val, binary_mask


def detect_and_filter_contours(
    binary_mask: np.ndarray,
    min_area: float
) -> tuple[int, int, list[float], np.ndarray]:
    """
    Finds external contours in the binary mask, filters them by minimum area,
    and returns (total_count, filtered_count, list_of_areas, contour_overlay_rgb).
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_objects = len(contours)
    passing_areas = []
    passing_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            passing_areas.append(area)
            passing_contours.append(contour)

    # Create an RGB visualization overlay of the detected/filtered objects
    overlay = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, passing_contours, -1, (0, 255, 0), 2)

    return total_objects, len(passing_contours), passing_areas, overlay


def compute_statistics(pixels: np.ndarray) -> dict:
    """Calculates basic statistical metrics for a pixel array."""
    if pixels.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    return {
        "mean": float(round(pixels.mean(), 2)),
        "std": float(round(pixels.std(), 2)),
        "min": int(pixels.min()),
        "max": int(pixels.max())
    }


def plot_histogram(data: np.ndarray, title: str, color: str = "steelblue") -> plt.Figure:
    """Generates a pixel intensity histogram figure (0-255)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(data.flatten(), bins=256, range=(0, 256), color=color, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel Brightness / Intensity", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=9)
    ax.set_xlim(0, 256)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# 2. Main Streamlit User Interface
# ------------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Medical Imaging Explorer",
        page_icon="🔬",
        layout="wide"
    )

    st.title("Medical Imaging Explorer 🔬")
    st.caption("Educational Image Processing & Computer Vision Lab | Built with OpenCV & Streamlit")

    # Educational Disclaimer Banner
    st.info(
        "ℹ️ **Educational Use Only**: This application is a computer vision learning project. "
        "It is strictly intended for educational exploration of algorithms (filtering, segmentation, and intensity distributions) "
        "and is **not** a diagnostic medical software."
    )

    # --------------------------------------------------------------------------
    # Sidebar: Controls & Inputs
    # --------------------------------------------------------------------------
    st.sidebar.header("📁 Image Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    sample_image_path = "sample_images/sample_chest_phantom.png"
    use_sample = False
    if uploaded_file is None:
        if os.path.exists(sample_image_path):
            use_sample = st.sidebar.checkbox("Load Sample Imaging Phantom", value=True)

    image = load_image(uploaded_file, sample_image_path if use_sample else None)

    if image is None:
        st.warning("👈 Please upload an image or check 'Load Sample Imaging Phantom' in the sidebar to get started.")
        return

    # Image Enhancement Controls
    with st.sidebar.expander("🎛️ Image Enhancements", expanded=True):
        brightness = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
        sharpen_strength = st.slider("Sharpen Strength", 0.0, 3.0, 0.0, 0.05)

    # Edge Detection & Segmentation Controls
    with st.sidebar.expander("🔍 Edges & Segmentation", expanded=True):
        detect_edges = st.checkbox("Enable Edge & Segmentation Pipeline", value=True)
        blur_strength = st.slider("Gaussian Blur Strength", 1, 15, 1, 2)
        auto_threshold = st.checkbox("Use Automatic Threshold (Otsu)", value=False)
        
        # Manual threshold slider is only shown/active when Otsu is disabled
        if not auto_threshold:
            manual_threshold = st.slider("Manual Threshold", 0, 255, 128, 1)
        else:
            manual_threshold = 128
            st.caption("✨ Otsu mode is active: threshold will be calculated automatically.")

        min_area = st.slider("Minimum Object Area", 0, 1000, 100, 5)

    # --------------------------------------------------------------------------
    # Apply Image Processing Pipeline
    # --------------------------------------------------------------------------
    # 1. Enhancements
    processed_image = apply_enhancements(image, brightness, contrast, sharpen_strength)
    gray_array = to_grayscale_array(processed_image)

    # 2. Gaussian Blur & Thresholding / Edges
    blurred_array = apply_gaussian_blur(gray_array, blur_strength)
    edges = detect_edges_canny(blurred_array, 100, 200)

    # Fixed bug: thresholding logic cleanly branches between Otsu and Manual
    active_threshold, binary_mask = apply_thresholding(
        blurred_array,
        use_otsu=auto_threshold,
        manual_threshold_val=manual_threshold
    )

    # 3. Contour Detection & Filtering
    total_objects, passed_objects, object_areas, contour_overlay = detect_and_filter_contours(
        binary_mask,
        min_area
    )

    # 4. White-Pixel Analysis
    white_pixels = int(np.sum(binary_mask == 255))
    total_pixels = int(binary_mask.size)
    white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # --------------------------------------------------------------------------
    # ROI Controls in Sidebar (bounded safely to image dimensions)
    # --------------------------------------------------------------------------
    img_height, img_width = gray_array.shape[:2]
    max_roi_dimension = max(20, min(img_width, img_height))
    default_roi_size = min(80, max_roi_dimension)

    with st.sidebar.expander("🎯 Region of Interest (ROI)", expanded=True):
        roi_size = st.slider(
            "ROI Size (pixels)",
            min_value=10,
            max_value=max_roi_dimension,
            value=default_roi_size,
            step=5
        )

        max_x = max(0, img_width - roi_size)
        max_y = max(0, img_height - roi_size)

        roi_x = st.slider("ROI X Position", 0, max_x, min(img_width // 4, max_x), 1)
        roi_y = st.slider("ROI Y Position", 0, max_y, min(img_height // 4, max_y), 1)

    # Extract ROI and draw rectangle
    roi_pixels = gray_array[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]

    image_with_roi = processed_image.copy()
    draw = ImageDraw.Draw(image_with_roi)
    draw.rectangle(
        [roi_x, roi_y, roi_x + roi_size, roi_y + roi_size],
        outline="red",
        width=3
    )

    # 5. Compute Statistics
    global_stats = compute_statistics(gray_array)
    roi_stats = compute_statistics(roi_pixels)

    # --------------------------------------------------------------------------
    # Main Tabs Organization
    # --------------------------------------------------------------------------
    tab_images, tab_segmentation, tab_roi, tab_stats = st.tabs([
        "🖼️ Image Enhancements",
        "🔬 Edges & Segmentation",
        "🎯 ROI Analysis",
        "📊 Statistics & Histograms"
    ])

    # --- TAB 1: Image Enhancements ---
    with tab_images:
        st.subheader("Visual Image Comparison")
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.image(image, caption=f"Original Image ({image.width} × {image.height})", use_container_width=True)
        with col_proc:
            st.image(processed_image, caption="Enhanced Image (Brightness / Contrast / Sharpen)", use_container_width=True)

        st.divider()
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        meta_col1.metric("Width", f"{image.width} px")
        meta_col2.metric("Height", f"{image.height} px")
        meta_col3.metric("Color Mode", image.mode)
        meta_col4.metric("OpenCV Version", cv2.__version__)

    # --- TAB 2: Edges & Segmentation ---
    with tab_segmentation:
        if not detect_edges:
            st.info("Enable 'Edge & Segmentation Pipeline' in the sidebar to view edge maps and binary masks.")
        else:
            st.subheader("Segmentation & Object Filtering")

            # Key segmentation metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Applied Threshold", f"{active_threshold}", help="Otsu calculated value or manual selection")
            col_m2.metric("Threshold Mode", "Otsu (Auto)" if auto_threshold else "Manual")
            col_m3.metric("Objects Detected", total_objects)
            col_m4.metric("Objects (Area ≥ Min)", passed_objects)

            st.write("")
            col_edge, col_bin, col_cont = st.columns(3)
            with col_edge:
                st.image(edges, caption="Canny Edges", use_container_width=True)
            with col_bin:
                st.image(binary_mask, caption=f"Binary Mask (Threshold: {active_threshold})", use_container_width=True)
            with col_cont:
                st.image(contour_overlay, caption=f"Filtered Contours (Area ≥ {min_area})", use_container_width=True)

            # White pixel percentage analysis
            st.divider()
            col_w1, col_w2 = st.columns([1, 2])
            with col_w1:
                st.metric("White Pixel Area", f"{round(white_percentage, 2)}%")
                st.caption(f"{white_pixels:,} white pixels out of {total_pixels:,} total pixels")
            with col_w2:
                with st.expander(f"📋 Filtered Object Area Details ({passed_objects} Objects)", expanded=False):
                    if passed_objects > 0:
                        df_objects = pd.DataFrame({
                            "Object ID": [f"Object {i+1}" for i in range(passed_objects)],
                            "Area (pixels²)": [round(a, 2) for a in object_areas]
                        })
                        st.dataframe(df_objects, use_container_width=True, hide_index=True)
                    else:
                        st.write("No objects met the minimum area filter threshold.")

    # --- TAB 3: ROI Analysis ---
    with tab_roi:
        st.subheader("Region of Interest (ROI) Inspection")
        col_full, col_zoom = st.columns([2, 1])

        with col_full:
            st.image(image_with_roi, caption=f"Full Image with ROI Box [X: {roi_x}, Y: {roi_y}, Size: {roi_size}px]", use_container_width=True)

        with col_zoom:
            roi_pil = Image.fromarray(roi_pixels)
            st.image(roi_pil, caption=f"Cropped ROI View ({roi_size} × {roi_size})", use_container_width=True)

            st.write("**ROI Local Intensity Summary:**")
            st.write(f"• Mean: {roi_stats['mean']}")
            st.write(f"• Std Dev: {roi_stats['std']}")
            st.write(f"• Min: {roi_stats['min']} | Max: {roi_stats['max']}")

    # --- TAB 4: Statistics & Histograms ---
    with tab_stats:
        st.subheader("Pixel Intensity Distribution & Statistics")

        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            st.markdown("#### 🌐 Global Image Statistics")
            g_c1, g_c2, g_c3, g_c4 = st.columns(4)
            g_c1.metric("Mean", global_stats["mean"])
            g_c2.metric("Std Dev", global_stats["std"])
            g_c3.metric("Min", global_stats["min"])
            g_c4.metric("Max", global_stats["max"])

            fig_global = plot_histogram(gray_array, "Global Image Pixel Intensity Histogram", color="#1f77b4")
            st.pyplot(fig_global)
            plt.close(fig_global)

        with stat_col2:
            st.markdown("#### 🎯 ROI Statistics")
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric("Mean", roi_stats["mean"])
            r_c2.metric("Std Dev", roi_stats["std"])
            r_c3.metric("Min", roi_stats["min"])
            r_c4.metric("Max", roi_stats["max"])

            fig_roi = plot_histogram(roi_pixels, f"ROI Pixel Intensity Histogram ({roi_size}x{roi_size})", color="#ff7f0e")
            st.pyplot(fig_roi)
            plt.close(fig_roi)


if __name__ == "__main__":
    main()
