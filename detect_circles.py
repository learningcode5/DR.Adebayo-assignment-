"""
Standalone OpenCV script to detect circular industrial components using cv2.HoughCircles.
Reads an image, applies grayscale conversion and Gaussian smoothing, detects circles
using the HOUGH_GRADIENT method, draws green circle outlines and red center dots,
and displays the result in standard OpenCV windows.
"""

from pathlib import Path
import cv2
import numpy as np

# ==============================================================================
# EASILY EDITABLE TUNING PARAMETERS (cv2.HoughCircles)
# ==============================================================================
MIN_DIST = 30       # Minimum distance between the centers of the detected circles
PARAM1 = 100        # Higher threshold of the two passed to the Canny edge detector
PARAM2 = 28         # Accumulator threshold for the circle centers at the detection stage
MIN_RADIUS = 8      # Minimum circle radius (in pixels)
MAX_RADIUS = 50     # Maximum circle radius (in pixels)

# Gaussian Blur settings (must be positive and odd)
BLUR_KERNEL_SIZE = (9, 9)
BLUR_SIGMA = 2.0

# Physical Calibration Reference Setting
REFERENCE_DIAMETER_MM = 28.0  # Physical diameter (mm) of the calibration anchor circle (first detected circle)

# Blue Table HSV Range & Perspective Warp Settings
APPLY_PERSPECTIVE = True
HSV_LOWER_BLUE = (85, 80, 60)
HSV_UPPER_BLUE = (130, 255, 255)
WARP_SIZE = 800
MIN_TABLE_AREA_RATIO = 0.15  # Minimum fraction of image area for automatic table detection

# Input image configuration (defaults to the first image found in the 'easy' or 'hard' folder)
DEFAULT_FOLDER = "hard"
IMAGE_PATH = ""  # Leave empty to automatically pick the first image in DEFAULT_FOLDER
# ==============================================================================


def order_corner_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner coordinates consistently as:
    [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # Sort by y-coordinate (top-to-bottom)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    # Sort top two by x-coordinate -> (top-left, top-right)
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    # Sort bottom two by x-coordinate -> (bottom-left, bottom-right)
    bl, br = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_blue_table_corners(
    image: np.ndarray,
    hsv_lower: tuple[int, int, int] = HSV_LOWER_BLUE,
    hsv_upper: tuple[int, int, int] = HSV_UPPER_BLUE,
    epsilon_factor: float = 0.02,
    min_area_ratio: float = MIN_TABLE_AREA_RATIO,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Isolate the blue table using cv2.cvtColor (to HSV) and cv2.inRange, find the largest
    contour, and use cv2.approxPolyDP to extract exactly four corner coordinates.

    Returns:
        (ordered_corners_4x2_or_None, hsv_mask)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))

    # Morphological cleanup so bolts near the border don't jaggedly break the table perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, cleaned_mask

    largest_contour = max(contours, key=cv2.contourArea)
    img_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest_contour) < min_area_ratio * img_area:
        return None, cleaned_mask

    # First try direct approxPolyDP on the largest contour with the requested epsilon
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon_factor * peri, True)
    if len(approx) == 4:
        return order_corner_points(approx.reshape(4, 2)), cleaned_mask

    # Fallback on the convex hull across a small epsilon sweep if border bolts indented the contour
    hull = cv2.convexHull(largest_contour)
    hull_peri = cv2.arcLength(hull, True)
    for eps in (epsilon_factor, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        approx_hull = cv2.approxPolyDP(hull, eps * hull_peri, True)
        if len(approx_hull) == 4:
            return order_corner_points(approx_hull.reshape(4, 2)), cleaned_mask

    return None, cleaned_mask


def warp_table_perspective(
    image: np.ndarray,
    corners: np.ndarray,
    warp_size: int = WARP_SIZE,
) -> np.ndarray:
    """
    Use cv2.getPerspectiveTransform to map the 4 ordered source points to a flat
    warp_size x warp_size (default 800x800) destination rectangle and apply cv2.warpPerspective.
    """
    src_pts = order_corner_points(corners)
    dst_pts = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (warp_size, warp_size))
    return warped


def resize_or_crop_to_square(
    image: np.ndarray,
    target_size: int = WARP_SIZE,
) -> np.ndarray:
    """
    Safely center-crop the raw image to a square aspect ratio (preserving circular
    geometry without stretching) and resize it to target_size x target_size (800x800)
    when perspective flattening is disabled or skipped.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")

    h, w = image.shape[:2]
    side = min(h, w)
    y_start = (h - side) // 2
    x_start = (w - side) // 2
    cropped = image[y_start : y_start + side, x_start : x_start + side]

    interp = cv2.INTER_AREA if side >= target_size else cv2.INTER_LINEAR
    return cv2.resize(cropped, (target_size, target_size), interpolation=interp)


def prepare_pipeline_image(
    image: np.ndarray,
    apply_perspective: bool = APPLY_PERSPECTIVE,
    corners: np.ndarray | None = None,
    hsv_lower: tuple[int, int, int] = HSV_LOWER_BLUE,
    hsv_upper: tuple[int, int, int] = HSV_UPPER_BLUE,
    epsilon_factor: float = 0.02,
    target_size: int = WARP_SIZE,
) -> tuple[np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    """
    Main preprocessing router before Hough Circle detection:
    - If apply_perspective is False: completely skips cv2.getPerspectiveTransform and
      cv2.warpPerspective, safely center-cropping and resizing the raw image to 800x800.
    - If apply_perspective is True: uses provided 4 corners or runs automatic HSV blue-table
      corner detection and warps to 800x800 (falling back to 800x800 resize/crop if no 4 corners exist).

    Returns:
        (prepared_800x800_bgr, mode_label, active_corners_or_None, hsv_mask_or_None)
    """
    if not apply_perspective:
        resized = resize_or_crop_to_square(image, target_size=target_size)
        return resized, f"Unwarped Resized/Cropped ({target_size}×{target_size})", None, None

    hsv_mask = None
    active_corners = None
    mode_label = ""

    if corners is not None and len(corners) == 4:
        active_corners = order_corner_points(np.asarray(corners, dtype=np.float32))
        mode_label = f"Manual 4 Corners ({target_size}×{target_size})"
    else:
        auto_corners, hsv_mask = detect_blue_table_corners(
            image,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            epsilon_factor=epsilon_factor,
        )
        if auto_corners is not None:
            active_corners = auto_corners
            mode_label = f"Auto HSV 4 Corners ({target_size}×{target_size})"

    if active_corners is not None:
        warped = warp_table_perspective(image, active_corners, warp_size=target_size)
        return warped, mode_label, active_corners, hsv_mask

    # Safe 800x800 fallback if 4 corners have not been clicked yet
    fallback_img = resize_or_crop_to_square(image, target_size=target_size)
    return (
        fallback_img,
        f"Fallback Resized/Cropped ({target_size}×{target_size})",
        None,
        hsv_mask,
    )


def resolve_image_path(custom_path: str, default_folder: str) -> Path:
    """Resolve the target image path to load."""
    base_dir = Path(__file__).resolve().parent
    if custom_path:
        p = Path(custom_path)
        return p if p.is_absolute() else base_dir / p

    folder_path = base_dir / default_folder
    if folder_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(
            [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )
        if images:
            return images[0]

    raise FileNotFoundError(
        f"No valid image found. Checked custom path '{custom_path}' and folder '{folder_path}'."
    )


def detect_and_draw_circles(
    image: np.ndarray,
    min_dist: float = MIN_DIST,
    param1: float = PARAM1,
    param2: float = PARAM2,
    min_radius: int = MIN_RADIUS,
    max_radius: int = MAX_RADIUS,
    blur_ksize: tuple[int, int] = BLUR_KERNEL_SIZE,
    blur_sigma: float = BLUR_SIGMA,
    reference_diameter_mm: float = REFERENCE_DIAMETER_MM,
) -> tuple[np.ndarray, int]:
    """
    Detect circular components in an image using HoughCircles, calibrate physical
    scale (pixels_per_mm) using the very first detected circle against
    reference_diameter_mm, and draw green circle outlines, red center dots, and
    yellow physical diameter labels (e.g., '28.3 mm').
    """
    annotated = image.copy()

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian smoothing to reduce noise and false circle detections
    blurred = cv2.GaussianBlur(gray, blur_ksize, blur_sigma)

    # 3. Detect circles using cv2.HoughCircles with HOUGH_GRADIENT
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    count = 0
    # 4. Calibrate using the first circle as anchor and draw outlines, centers, and mm labels
    if circles is not None:
        raw_circles = circles[0, :]
        circles_rounded = np.uint16(np.around(circles))
        count = len(circles_rounded[0, :])

        # Use the very first detected circle in the array as the calibration anchor
        anchor_radius_px = float(raw_circles[0][2])
        anchor_diameter_px = anchor_radius_px * 2.0
        ref_mm = float(reference_diameter_mm) if reference_diameter_mm > 0 else 28.0
        pixels_per_mm = (anchor_diameter_px / ref_mm) if anchor_diameter_px > 0 else 1.0

        img_h, img_w = annotated.shape[:2]

        for idx, (x, y, r) in enumerate(circles_rounded[0, :]):
            ix, iy, ir = int(x), int(y), int(r)
            # Green circle outline: BGR (0, 255, 0)
            cv2.circle(annotated, (ix, iy), ir, (0, 255, 0), 2)
            # Red center dot: BGR (0, 0, 255)
            cv2.circle(annotated, (ix, iy), 3, (0, 0, 255), -1)

            # Calculate physical diameter in mm using the unrounded float radius (or pixel diameter)
            circle_diameter_px = float(raw_circles[idx][2]) * 2.0
            physical_diameter_mm = circle_diameter_px / pixels_per_mm if pixels_per_mm > 0 else 0.0
            label_text = f"{physical_diameter_mm:.1f} mm"

            # Position text directly next to the green circle (clamped inside image bounds)
            text_x = min(max(5, ix + ir + 6), max(5, img_w - 110))
            text_y = min(max(20, iy + 5), max(20, img_h - 10))

            # Draw dark outline first for readability, then yellow text: BGR (0, 255, 255)
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return annotated, count


def main() -> None:
    target_path = resolve_image_path(IMAGE_PATH, DEFAULT_FOLDER)
    print(f"Loading image: {target_path}")

    original_img = cv2.imread(str(target_path))
    if original_img is None:
        raise ValueError(f"Failed to read image at {target_path}")

    source_img, mode_label, _, _ = prepare_pipeline_image(
        original_img,
        apply_perspective=APPLY_PERSPECTIVE,
        target_size=WARP_SIZE,
    )
    print(f"Prepared 800x800 pipeline image using mode: {mode_label}")

    annotated_img, num_circles = detect_and_draw_circles(
        source_img,
        min_dist=MIN_DIST,
        param1=PARAM1,
        param2=PARAM2,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS,
        reference_diameter_mm=REFERENCE_DIAMETER_MM,
    )

    print(
        f"Detected {num_circles} circular component(s) "
        f"[minDist={MIN_DIST}, param1={PARAM1}, param2={PARAM2}, "
        f"minRadius={MIN_RADIUS}, maxRadius={MAX_RADIUS}]"
    )

    # Standard OpenCV display windows
    cv2.namedWindow("Original / Bird's-Eye Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detected Circular Components", cv2.WINDOW_NORMAL)

    cv2.imshow("Original / Bird's-Eye Image", source_img)
    cv2.imshow("Detected Circular Components", annotated_img)

    print("Press any key in the OpenCV image window (or close the window) to exit...")
    while True:
        key = cv2.waitKey(100)
        if key != -1:
            break
        # Exit cleanly if the user clicks the 'X' button on either window
        if (
            cv2.getWindowProperty("Original / Bird's-Eye Image", cv2.WND_PROP_VISIBLE) < 1
            or cv2.getWindowProperty("Detected Circular Components", cv2.WND_PROP_VISIBLE) < 1
        ):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
