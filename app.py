"""
Medical Imaging Explorer
An image processing and analysis tool built with Python, OpenCV, and Streamlit.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt
import streamlit as st


def load_image(uploaded_file, sample_path=None) -> Image.Image:
    """Load an image from upload or file path and convert to RGB."""
    if uploaded_file is not None:
        return Image.open(uploaded_file).convert("RGB")
    elif sample_path and os.path.exists(sample_path):
        return Image.open(sample_path).convert("RGB")
    return None


def apply_enhancements(
    pil_image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpen_strength: float = 0.0
) -> Image.Image:
    """Adjust brightness, contrast, and apply Laplacian sharpening."""
    enhanced = ImageEnhance.Brightness(pil_image).enhance(brightness)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)

    if sharpen_strength > 0:
        img_array = np.array(enhanced)
        center = 5.0 + sharpen_strength
        kernel = np.array([
            [0, -1, 0],
            [-1, center, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        sharpened_arr = cv2.filter2D(img_array, -1, kernel)
        enhanced = Image.fromarray(sharpened_arr)

    return enhanced


def to_grayscale_array(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to 8-bit grayscale array."""
    return np.array(pil_image.convert("L"))


def apply_gaussian_blur(gray_array: np.ndarray, blur_strength: int) -> np.ndarray:
    """Apply Gaussian blur with odd kernel size."""
    if blur_strength > 1:
        ksize = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
        return cv2.GaussianBlur(gray_array, (ksize, ksize), 0)
    return gray_array


def detect_edges_canny(
    gray_array: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> np.ndarray:
    """Apply Canny edge detection."""
    return cv2.Canny(gray_array, low_threshold, high_threshold)


def apply_thresholding(
    gray_array: np.ndarray,
    use_otsu: bool = False,
    manual_threshold_val: int = 128
) -> tuple[int, np.ndarray]:
    """Apply Otsu or manual binary thresholding."""
    if use_otsu:
        computed_thresh, binary_mask = cv2.threshold(
            gray_array,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return int(computed_thresh), binary_mask
    else:
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
    """Find contours, filter by minimum area, and return overlay mask."""
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

    overlay = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, passing_contours, -1, (0, 255, 0), 2)

    return total_objects, len(passing_contours), passing_areas, overlay


def compute_statistics(pixels: np.ndarray) -> dict:
    """Calculate basic statistical values for pixel array."""
    if pixels.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    return {
        "mean": float(round(pixels.mean(), 2)),
        "std": float(round(pixels.std(), 2)),
        "min": int(pixels.min()),
        "max": int(pixels.max())
    }


def plot_histogram(data: np.ndarray, title: str, color: str = "#4a5568") -> plt.Figure:
    """Generate pixel intensity histogram."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(data.flatten(), bins=256, range=(0, 256), color=color, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Pixel Intensity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlim(0, 256)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Medical Imaging Explorer",
        layout="wide"
    )

    st.title("Medical Imaging Explorer")
    st.caption("Image processing and analysis tool built with OpenCV and Streamlit")

    # --------------------------------------------------------------------------
    # Static Example / Illustration Section
    # --------------------------------------------------------------------------
    with st.expander("Example Pipeline Stages", expanded=True):
        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
        with col_ex1:
            st.image("examples/01_original_xray.png", caption="Original X-ray", use_container_width=True)
        with col_ex2:
            st.image("examples/02_canny_edges.png", caption="Canny Edges", use_container_width=True)
        with col_ex3:
            st.image("examples/03_thresholded_image.png", caption="Thresholded Image", use_container_width=True)
        with col_ex4:
            st.image("examples/04_contours.png", caption="Contours", use_container_width=True)

    # --------------------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------------------
    st.sidebar.header("Image Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    example_xray_path = "examples/01_original_xray.png"
    use_example = False
    if uploaded_file is None and os.path.exists(example_xray_path):
        use_example = st.sidebar.checkbox("Use example X-ray for live processing", value=True)

    # Global toggle for optional explanatory help text
    show_explanations = st.sidebar.checkbox("Show explanations", value=False)

    image = load_image(uploaded_file, example_xray_path if use_example else None)

    if image is None:
        st.info("Upload an image or check 'Use example X-ray for live processing' in the sidebar to run the live pipeline.")
        return

    with st.sidebar.expander("Image Adjustments", expanded=True):
        brightness = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
        if show_explanations:
            st.caption(
                "Adjusts the overall intensity of the image. "
                "It is useful when the image is too dark or too bright and details are difficult to see. "
                "Try small changes first and compare the result with the original."
            )

        contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
        if show_explanations:
            st.caption(
                "Changes the difference between dark and bright regions. "
                "It is useful when structures blend together because the image has low contrast. "
                "Increase it gradually and watch whether important intensity differences become easier to see."
            )

        sharpen_strength = st.slider("Sharpen", 0.0, 3.0, 0.0, 0.05)
        if show_explanations:
            st.caption(
                "Emphasizes fine details and edges. "
                "It is useful when the image appears slightly soft or boundaries are difficult to see. "
                "Use moderate values because strong sharpening can also make noise and unwanted edges more visible."
            )

    with st.sidebar.expander("Segmentation", expanded=True):
        detect_edges = st.checkbox("Enable segmentation pipeline", value=True)
        if show_explanations:
            st.caption(
                "Runs the full segmentation pipeline: Gaussian blur, Canny edge detection, thresholding, and contour analysis. "
                "Disable to view only the adjusted image without segmentation output."
            )

        blur_strength = st.slider("Gaussian blur", 1, 15, 1, 2)
        if show_explanations:
            st.caption(
                "Smooths the image and reduces small intensity variations. "
                "It is useful before edge detection when noise creates many unwanted edges. "
                "Try a small blur first; too much smoothing can remove fine details."
            )

        auto_threshold = st.checkbox("Use Otsu thresholding", value=False)
        if show_explanations:
            st.caption(
                "Automatically selects a global threshold from the image intensity distribution. "
                "It is useful when you are not sure which manual threshold to start with. "
                "Compare the Otsu result with manual thresholding to see how the selected regions change."
            )

        if not auto_threshold:
            manual_threshold = st.slider("Threshold", 0, 255, 128, 1)
            if show_explanations:
                st.caption(
                    "Separates pixels according to their intensity and produces a binary image. "
                    "Pixels above the threshold become white; pixels below become black. "
                    "Move the threshold gradually and watch how the selected regions change."
                )
        else:
            manual_threshold = 128
            st.caption("Otsu thresholding active")

        min_area = st.slider("Minimum object area", 0, 1000, 100, 5)
        if show_explanations:
            st.caption(
                "Removes detected regions smaller than the selected area. "
                "It is useful when thresholding produces many tiny unwanted regions. "
                "Increase the value to remove more small regions, or reduce it when smaller regions need to be kept."
            )

    # Image Processing Pipeline
    processed_image = apply_enhancements(image, brightness, contrast, sharpen_strength)
    gray_array = to_grayscale_array(processed_image)

    blurred_array = apply_gaussian_blur(gray_array, blur_strength)
    edges = detect_edges_canny(blurred_array, 100, 200)

    active_threshold, binary_mask = apply_thresholding(
        blurred_array,
        use_otsu=auto_threshold,
        manual_threshold_val=manual_threshold
    )

    total_objects, passed_objects, object_areas, contour_overlay = detect_and_filter_contours(
        binary_mask,
        min_area
    )

    white_pixels = int(np.sum(binary_mask == 255))
    total_pixels = int(binary_mask.size)
    white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # ROI Parameters
    img_height, img_width = gray_array.shape[:2]
    max_roi_dimension = max(20, min(img_width, img_height))
    default_roi_size = min(80, max_roi_dimension)

    with st.sidebar.expander("ROI", expanded=True):
        roi_size = st.slider(
            "ROI size",
            min_value=10,
            max_value=max_roi_dimension,
            value=default_roi_size,
            step=5
        )
        if show_explanations:
            st.caption(
                "Selects a smaller region of the image for focused analysis. "
                "It is useful when the whole image contains areas that are not relevant to the measurement you want to make. "
                "For example, select one region and compare its statistics with the full image."
            )

        max_x = max(0, img_width - roi_size)
        max_y = max(0, img_height - roi_size)

        roi_x = st.slider("ROI X position", 0, max_x, min(img_width // 4, max_x), 1)
        roi_y = st.slider("ROI Y position", 0, max_y, min(img_height // 4, max_y), 1)
        if show_explanations:
            st.caption("Horizontal and vertical position of the top-left corner of the ROI box.")


    roi_pixels = gray_array[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]

    image_with_roi = processed_image.copy()
    draw = ImageDraw.Draw(image_with_roi)
    draw.rectangle(
        [roi_x, roi_y, roi_x + roi_size, roi_y + roi_size],
        outline="red",
        width=2
    )

    global_stats = compute_statistics(gray_array)
    roi_stats = compute_statistics(roi_pixels)

    # --------------------------------------------------------------------------
    # Main Tabs for Live Processing Outputs
    # --------------------------------------------------------------------------
    tab_images, tab_segmentation, tab_roi, tab_stats = st.tabs([
        "Images",
        "Segmentation",
        "ROI",
        "Statistics"
    ])

    # --- Tab 1: Images ---
    with tab_images:
        st.subheader("Live Images")
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.image(image, caption="Current Input Image", use_container_width=True)
        with col_proc:
            st.image(processed_image, caption="Processed Image", use_container_width=True)

        st.caption(f"Dimensions: {image.width}x{image.height} | Mode: {image.mode} | OpenCV version: {cv2.__version__}")

    # --- Tab 2: Segmentation ---
    with tab_segmentation:
        if not detect_edges:
            st.info("Segmentation pipeline is disabled. Enable it in the sidebar.")
        else:
            st.subheader("Live Segmentation")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Threshold", active_threshold)
            m2.metric("Threshold Mode", "Otsu" if auto_threshold else "Manual")
            m3.metric("Detected Objects", total_objects)
            m4.metric("Filtered Objects", passed_objects)

            st.write("")
            col_edge, col_bin, col_cont = st.columns(3)
            with col_edge:
                st.image(edges, caption="Canny Edges", use_container_width=True)
                if show_explanations:
                    st.caption(
                        "Highlights boundaries where image intensity changes rapidly. "
                        "It is useful when you want to inspect structural edges in the image. "
                        "If the result contains too many small unwanted edges, try applying more smoothing before Canny."
                    )
            with col_bin:
                st.image(binary_mask, caption=f"Thresholded Image (Threshold: {active_threshold})", use_container_width=True)
            with col_cont:
                st.image(contour_overlay, caption=f"Contours (Area >= {min_area})", use_container_width=True)
                if show_explanations:
                    st.caption(
                        "Finds boundaries of connected regions in the processed image. "
                        "It is useful when you want to identify or measure detected regions after thresholding. "
                        "The number and size of detected contours depend on the quality of the preceding image processing."
                    )

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("White Pixels", f"{round(white_percentage, 2)}%")
                st.caption(f"{white_pixels:,} / {total_pixels:,} pixels")
            with c2:
                with st.expander(f"Detected Object Areas ({passed_objects})", expanded=False):
                    if passed_objects > 0:
                        df_objects = pd.DataFrame({
                            "Object": [f"Object {i+1}" for i in range(passed_objects)],
                            "Area": [round(a, 2) for a in object_areas]
                        })
                        st.dataframe(df_objects, use_container_width=True, hide_index=True)
                    else:
                        st.write("No objects found with area above the minimum threshold.")

    # --- Tab 3: ROI ---
    with tab_roi:
        st.subheader("Live ROI")
        col_full, col_zoom = st.columns([2, 1])

        with col_full:
            st.image(image_with_roi, caption=f"Selected ROI (x={roi_x}, y={roi_y}, size={roi_size})", use_container_width=True)

        with col_zoom:
            roi_pil = Image.fromarray(roi_pixels)
            st.image(roi_pil, caption="ROI Crop", use_container_width=True)

            st.write("ROI Statistics:")
            st.text(f"Mean: {roi_stats['mean']}\nStd:  {roi_stats['std']}\nMin:  {roi_stats['min']}\nMax:  {roi_stats['max']}")

    # --- Tab 4: Statistics ---
    with tab_stats:
        st.subheader("Live Statistics")

        if show_explanations:
            st.caption(
                "Summarizes pixel intensity values in the selected region. "
                "It is useful when you want to compare image regions numerically rather than relying only on visual inspection. "
                "Compare the ROI statistics with the corresponding values for the full image."
            )

        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            st.write("**Global Statistics**")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Mean", global_stats["mean"])
            g2.metric("Std", global_stats["std"])
            g3.metric("Min", global_stats["min"])
            g4.metric("Max", global_stats["max"])

            fig_global = plot_histogram(gray_array, "Global Intensity Histogram", color="#4a5568")
            st.pyplot(fig_global)
            plt.close(fig_global)
            if show_explanations:
                st.caption(
                    "Shows how pixel intensities are distributed across the entire image. "
                    "It is useful for understanding brightness, contrast, and whether intensity groups appear separated. "
                    "The histogram can also help when choosing a manual threshold."
                )

        with stat_col2:
            st.write("**ROI Statistics**")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Mean", roi_stats["mean"])
            r2.metric("Std", roi_stats["std"])
            r3.metric("Min", roi_stats["min"])
            r4.metric("Max", roi_stats["max"])

            fig_roi = plot_histogram(roi_pixels, "ROI Intensity Histogram", color="#718096")
            st.pyplot(fig_roi)
            plt.close(fig_roi)
            if show_explanations:
                st.caption(
                    "Shows how pixel intensities are distributed within the selected region. "
                    "Compare it with the global histogram to see how the selected area differs from the whole image."
                )



if __name__ == "__main__":
    main()
