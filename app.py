"""
Local Web Interface for Circular Industrial Component Detection (OpenCV HoughCircles).

Features:
- Visual Thumbnail Gallery (inside a sidebar expander AND an optional main gallery tab)
  replacing the text-only image filename dropdown, allowing visual dataset browsing by clutter.
- Perspective Transformation preprocessing step:
  1. Isolates the blue table via cv2.cvtColor (to HSV) and cv2.inRange.
  2. Finds the largest contour and uses cv2.approxPolyDP to extract 4 corner coordinates.
  3. Manual fallback using streamlit-image-coordinates so the user can click the 4 table corners
     directly on the UI whenever automated detection fails (or when manual override is toggled).
  4. Maps the 4 source corners to a flat 800x800 destination rectangle with cv2.getPerspectiveTransform
     and cv2.warpPerspective, routing the flattened bird's-eye view into the Hough Circles pipeline.
- Interactive Hough Circle sliders, session state evaluation logging, and CSV export.
"""

import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import detect_circles

# Reload detect_circles on each Streamlit rerun so newly added functions/constants are never stale in sys.modules
importlib.reload(detect_circles)

from detect_circles import (
    APPLY_PERSPECTIVE,
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA,
    HSV_LOWER_BLUE,
    HSV_UPPER_BLUE,
    MAX_RADIUS,
    MIN_DIST,
    MIN_RADIUS,
    PARAM1,
    PARAM2,
    REFERENCE_DIAMETER_MM,
    WARP_SIZE,
    detect_and_draw_circles,
    detect_blue_table_corners,
    order_corner_points,
    prepare_pipeline_image,
    resize_or_crop_to_square,
    warp_table_perspective,
)

BASE_DIR = Path(__file__).resolve().parent
LABELS_FILE = BASE_DIR / "folder_labels.json"
REVIEW_QUEUE_DIR = BASE_DIR / "review_queue"
BATCH_METADATA_FILE = REVIEW_QUEUE_DIR / "batch_metadata.json"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONDITION_OPTIONS = ["Easy", "Moderate", "Difficult"]
EXCLUDED_DIRS = {".venv", "venv", "__pycache__", ".git", ".idea", ".vscode", "review_queue"}


def natural_sort_key(name: str) -> list:
    """Sort strings containing numbers in natural numerical order (e.g. easy2.jpg before easy10.jpg)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_folder_labels() -> dict[str, str]:
    """Load saved condition labels for folders from disk."""
    if LABELS_FILE.is_file():
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_folder_labels(labels: dict[str, str]) -> None:
    """Persist folder condition labels to disk."""
    try:
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    except OSError:
        pass


def load_batch_metadata() -> dict[str, dict]:
    """Load saved metadata for annotated images in the review_queue folder."""
    if BATCH_METADATA_FILE.is_file():
        try:
            with open(BATCH_METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_batch_metadata(meta: dict[str, dict]) -> None:
    """Persist metadata for annotated images in the review_queue folder."""
    REVIEW_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(BATCH_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass


def scan_local_folders(base_dir: Path) -> list[str]:
    """Scan the local project directory for image folders."""
    folders = []
    for entry in sorted(base_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in EXCLUDED_DIRS:
            folders.append(entry.name)
    return folders


def list_images_in_folder(folder_path: Path) -> list[str]:
    """Return a naturally sorted list of image filenames inside the selected folder."""
    if not folder_path.is_dir():
        return []
    return sorted(
        [
            f.name
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS
        ],
        key=natural_sort_key,
    )


def infer_default_condition(folder_name: str, saved_labels: dict[str, str]) -> str:
    """Infer the default condition label for a folder based on saved state or folder name."""
    if folder_name in saved_labels and saved_labels[folder_name] in CONDITION_OPTIONS:
        return saved_labels[folder_name]
    lower_name = folder_name.lower()
    if "hard" in lower_name or "difficult" in lower_name:
        return "Difficult"
    if "moderate" in lower_name or "medium" in lower_name:
        return "Moderate"
    return "Easy"


@st.cache_data(show_spinner=False)
def load_thumbnail_rgb(image_path_str: str, thumb_width: int = 160) -> np.ndarray | None:
    """Read an image from disk and resize it to a compact RGB thumbnail for the gallery."""
    bgr = cv2.imread(image_path_str)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = thumb_width / float(w)
    new_h = max(1, int(round(h * scale)))
    resized_bgr = cv2.resize(bgr, (thumb_width, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)


def draw_corner_overlay(image_bgr: np.ndarray, corners: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    """Draw detected or clicked corners (and polygon outline if 4 points exist) on a BGR image copy."""
    vis = image_bgr.copy()
    pts = [tuple(map(int, map(round, pt))) for pt in corners]
    if len(pts) == 4:
        ordered = order_corner_points(np.array(pts, dtype=np.float32)).astype(np.int32)
        cv2.polylines(vis, [ordered.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 255), thickness=4)
        labels = ["TL", "TR", "BR", "BL"]
        for idx, (x, y) in enumerate(ordered):
            cv2.circle(vis, (x, y), 12, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 14, (255, 255, 255), 2)
            cv2.putText(
                vis,
                labels[idx],
                (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
    else:
        for idx, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 12, (0, 165, 255), -1)
            cv2.circle(vis, (x, y), 14, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f"P{idx + 1}",
                (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
    return vis


def update_detect_circles_script(
    min_dist: int,
    param1: int,
    param2: int,
    min_radius: int,
    max_radius: int,
    blur_k: int,
    blur_sigma: float,
    apply_perspective: bool,
    reference_diameter_mm: float,
    folder_name: str,
    image_rel_path: str,
) -> None:
    """Sync the tuned parameters and selected image back into detect_circles.py."""
    script_path = BASE_DIR / "detect_circles.py"
    if not script_path.is_file():
        return
    content = script_path.read_text(encoding="utf-8")
    replacements = {
        r"^MIN_DIST\s*=.*": f"MIN_DIST = {min_dist}       # Minimum distance between the centers of the detected circles",
        r"^PARAM1\s*=.*": f"PARAM1 = {param1}        # Higher threshold of the two passed to the Canny edge detector",
        r"^PARAM2\s*=.*": f"PARAM2 = {param2}         # Accumulator threshold for the circle centers at the detection stage",
        r"^MIN_RADIUS\s*=.*": f"MIN_RADIUS = {min_radius}      # Minimum circle radius (in pixels)",
        r"^MAX_RADIUS\s*=.*": f"MAX_RADIUS = {max_radius}     # Maximum circle radius (in pixels)",
        r"^BLUR_KERNEL_SIZE\s*=.*": f"BLUR_KERNEL_SIZE = ({blur_k}, {blur_k})",
        r"^BLUR_SIGMA\s*=.*": f"BLUR_SIGMA = {blur_sigma}",
        r"^REFERENCE_DIAMETER_MM\s*=.*": f"REFERENCE_DIAMETER_MM = {reference_diameter_mm}",
        r"^APPLY_PERSPECTIVE\s*=.*": f"APPLY_PERSPECTIVE = {apply_perspective}",
        r"^DEFAULT_FOLDER\s*=.*": f'DEFAULT_FOLDER = "{folder_name}"',
        r"^IMAGE_PATH\s*=.*": f'IMAGE_PATH = "{image_rel_path}"  # Selected image to inspect in OpenCV windows',
    }
    for pattern, repl in replacements.items():
        content = re.sub(pattern, repl, content, flags=re.MULTILINE)
    script_path.write_text(content, encoding="utf-8")


def render_thumbnail_gallery(
    folder_path: Path,
    image_files: list[str],
    selected_image_name: str,
    cols_per_row: int,
    key_prefix: str,
) -> None:
    """Render a paginated thumbnail gallery grid using st.columns with Select buttons."""
    page_size = 18
    total_images = len(image_files)
    total_pages = max(1, (total_images + page_size - 1) // page_size)

    # Default page to the page containing the currently selected image
    current_idx = image_files.index(selected_image_name) if selected_image_name in image_files else 0
    default_page = (current_idx // page_size) + 1

    if total_pages > 1:
        page = st.number_input(
            f"Gallery Page (1–{total_pages}, {total_images} images)",
            min_value=1,
            max_value=total_pages,
            value=default_page,
            step=1,
            key=f"{key_prefix}_page",
        )
    else:
        page = 1

    start_i = (page - 1) * page_size
    end_i = min(total_images, start_i + page_size)
    subset = image_files[start_i:end_i]

    for row_start in range(0, len(subset), cols_per_row):
        row_files = subset[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, fname in enumerate(row_files):
            global_idx = start_i + row_start + col_idx
            img_full_path = folder_path / fname
            thumb = load_thumbnail_rgb(str(img_full_path), thumb_width=180)
            is_selected = fname == selected_image_name
            short_label = fname.split(".rf.")[0] if ".rf." in fname else fname[:18]

            with cols[col_idx]:
                if thumb is not None:
                    st.image(thumb, caption=f"#{global_idx + 1}: {short_label}", use_container_width=True)
                btn_label = "✓ Active" if is_selected else "Select"
                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    btn_label,
                    key=f"{key_prefix}_sel_{global_idx}_{fname}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state["selected_image_by_folder"][folder_path.name] = fname
                    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Industrial Circle Detector - Parameter Tuner",
        layout="wide",
    )

    st.title("Circular Industrial Component Detector (`cv2.HoughCircles`)")
    st.markdown(
        "Browse images visually from the thumbnail gallery, flatten the table into an **800×800 bird's-eye view** "
        "(automatically via HSV contour approximation or via interactive 4-corner clicking), and tune `HOUGH_GRADIENT` parameters."
    )

    # Initialize session state structures
    if "results_log" not in st.session_state:
        st.session_state["results_log"] = []
    if "selected_image_by_folder" not in st.session_state:
        st.session_state["selected_image_by_folder"] = {}
    if "manual_corners" not in st.session_state:
        st.session_state["manual_corners"] = {}
    if "last_click_coord" not in st.session_state:
        st.session_state["last_click_coord"] = {}
    if "review_index" not in st.session_state:
        st.session_state["review_index"] = 0

    folder_labels = load_folder_labels()

    # --------------------------------------------------------------------------
    # Sidebar: 1. Folder Selection, Condition Label & Thumbnail Gallery
    # --------------------------------------------------------------------------
    st.sidebar.header("1. Dataset & Visual Gallery")

    folders = scan_local_folders(BASE_DIR)
    if not folders:
        st.error(
            f"No image folders found in `{BASE_DIR}`. Please create a folder (e.g., `Easy`) "
            "and place your images inside it."
        )
        return

    default_folder_idx = 0
    for idx, name in enumerate(folders):
        if name.lower() == "easy":
            default_folder_idx = idx
            break

    selected_folder = st.sidebar.selectbox(
        "Select Local Folder",
        options=folders,
        index=default_folder_idx,
        help="Automatically scans the local project directory for image folders.",
    )

    default_condition = infer_default_condition(selected_folder, folder_labels)
    condition_idx = CONDITION_OPTIONS.index(default_condition)

    selected_condition = st.sidebar.selectbox(
        f"Condition Label for '{selected_folder}'",
        options=CONDITION_OPTIONS,
        index=condition_idx,
        help="Assign a difficulty/condition label (Easy, Moderate, Difficult) to the active folder.",
    )

    if folder_labels.get(selected_folder) != selected_condition:
        folder_labels[selected_folder] = selected_condition
        save_folder_labels(folder_labels)

    folder_path = BASE_DIR / selected_folder
    image_files = list_images_in_folder(folder_path)

    if not image_files:
        st.warning(f"No valid image files found inside `{selected_folder}/`.")
        return

    # Determine currently active image in the selected folder
    saved_img = st.session_state["selected_image_by_folder"].get(selected_folder)
    if saved_img not in image_files:
        saved_img = image_files[0]
        st.session_state["selected_image_by_folder"][selected_folder] = saved_img
    selected_image_name = saved_img

    cur_idx = image_files.index(selected_image_name)
    st.sidebar.caption(f"**Active Image ({cur_idx + 1}/{len(image_files)}):** `{selected_image_name}`")

    # Quick Prev / Next buttons alongside the Visual Thumbnail Gallery
    nav_prev, nav_next = st.sidebar.columns(2)
    with nav_prev:
        if st.button("◀ Previous", use_container_width=True, disabled=(cur_idx == 0)):
            st.session_state["selected_image_by_folder"][selected_folder] = image_files[cur_idx - 1]
            st.rerun()
    with nav_next:
        if st.button("Next ▶", use_container_width=True, disabled=(cur_idx == len(image_files) - 1)):
            st.session_state["selected_image_by_folder"][selected_folder] = image_files[cur_idx + 1]
            st.rerun()

    with st.sidebar.expander(f"🖼 Visual Thumbnail Gallery ({len(image_files)} images)", expanded=True):
        st.caption("Click **Select** under any thumbnail to load it into the main tuning interface.")
        render_thumbnail_gallery(
            folder_path=folder_path,
            image_files=image_files,
            selected_image_name=selected_image_name,
            cols_per_row=2,
            key_prefix="sidebar_gal",
        )

    # --------------------------------------------------------------------------
    # Sidebar: 2. Perspective Transform Settings (Blue Table HSV + Warp)
    # --------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("2. Perspective Transform (800×800)")
    enable_perspective = st.sidebar.checkbox(
        "Apply Perspective Flattening",
        value=bool(APPLY_PERSPECTIVE),
        help=(
            "When checked, isolates the table and warps the 4 corners to an 800×800 bird's-eye view. "
            "When unchecked, completely skips cv2.getPerspectiveTransform and cv2.warpPerspective, "
            "safely cropping/resizing the raw image to 800×800."
        ),
    )
    force_manual_corners = st.sidebar.checkbox(
        "Force Manual 4-Corner Selection",
        value=False,
        disabled=(not enable_perspective),
        help="Check this to manually click the 4 table corners even if automatic HSV detection succeeded.",
    )

    with st.sidebar.expander("Blue Table HSV & Polygon Tuning", expanded=False):
        h_min, h_max = st.slider("Hue Range (Blue)", 0, 179, (HSV_LOWER_BLUE[0], HSV_UPPER_BLUE[0]))
        s_min, s_max = st.slider("Saturation Range", 0, 255, (HSV_LOWER_BLUE[1], HSV_UPPER_BLUE[1]))
        v_min, v_max = st.slider("Value Range", 0, 255, (HSV_LOWER_BLUE[2], HSV_UPPER_BLUE[2]))
        eps_factor = st.slider("approxPolyDP Epsilon Factor", 0.005, 0.080, 0.020, step=0.005)

    # --------------------------------------------------------------------------
    # Sidebar: 3. Interactive Hough Circle Sliders
    # --------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("3. Hough Circle Parameters")

    min_dist = st.sidebar.slider(
        "minDist (Min distance between centers)",
        min_value=1,
        max_value=500,
        value=int(MIN_DIST),
        step=1,
    )
    param1 = st.sidebar.slider(
        "param1 (Canny high threshold)",
        min_value=10,
        max_value=300,
        value=int(PARAM1),
        step=1,
    )
    param2 = st.sidebar.slider(
        "param2 (Accumulator threshold)",
        min_value=5,
        max_value=150,
        value=int(PARAM2),
        step=1,
    )
    min_radius = st.sidebar.slider(
        "minRadius (Minimum circle radius)",
        min_value=0,
        max_value=300,
        value=int(MIN_RADIUS),
        step=1,
    )
    max_radius = st.sidebar.slider(
        "maxRadius (Maximum circle radius)",
        min_value=1,
        max_value=500,
        value=int(MAX_RADIUS),
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Gaussian Smoothing")
    blur_k = st.sidebar.slider(
        "Gaussian Kernel Size (odd)",
        min_value=3,
        max_value=31,
        value=int(BLUR_KERNEL_SIZE[0]),
        step=2,
    )
    blur_sigma = st.sidebar.slider(
        "Gaussian Sigma",
        min_value=0.5,
        max_value=10.0,
        value=float(BLUR_SIGMA),
        step=0.5,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Physical Size Calibration")
    reference_diameter_mm = st.sidebar.number_input(
        "Reference Object Physical Diameter (mm)",
        min_value=0.1,
        max_value=1000.0,
        value=float(REFERENCE_DIAMETER_MM),
        step=0.5,
        help="Physical diameter (mm) of the calibration anchor (first detected circle).",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("4. Standalone OpenCV Script")
    rel_img_path = f"{selected_folder}/{selected_image_name}"

    if st.sidebar.button("Save & Run detect_circles.py (OpenCV Windows)", use_container_width=True):
        update_detect_circles_script(
            min_dist=min_dist,
            param1=param1,
            param2=param2,
            min_radius=min_radius,
            max_radius=max_radius,
            blur_k=blur_k,
            blur_sigma=blur_sigma,
            apply_perspective=enable_perspective,
            reference_diameter_mm=reference_diameter_mm,
            folder_name=selected_folder,
            image_rel_path=rel_img_path,
        )
        subprocess.Popen([sys.executable, str(BASE_DIR / "detect_circles.py")], cwd=str(BASE_DIR))
        st.sidebar.success("Saved parameters to `detect_circles.py` and opened OpenCV windows!")

    # --------------------------------------------------------------------------
    # Main Area Tabs: Tuning & Inspection, Batch Processing, and Visual Gallery
    # --------------------------------------------------------------------------
    tab_tune, tab_batch, tab_gallery = st.tabs(
        [
            "🎯 Parameter Tuning & Perspective Warp",
            "⚙️ Batch Processing & Review Mode",
            "🖼 Full-Screen Dataset Gallery",
        ]
    )

    with tab_gallery:
        st.subheader(f"Visual Thumbnail Browser — `{selected_folder}` ({len(image_files)} images)")
        st.markdown(
            "Visually inspect the table layout, surface color, and component clutter across the folder. "
            "Click **Select** under any image to load it into the **Parameter Tuning & Perspective Warp** tab."
        )
        render_thumbnail_gallery(
            folder_path=folder_path,
            image_files=image_files,
            selected_image_name=selected_image_name,
            cols_per_row=6,
            key_prefix="main_gal",
        )

    with tab_tune:
        image_path = folder_path / selected_image_name
        bgr_image = cv2.imread(str(image_path))

        if bgr_image is None:
            st.error(f"Could not load image at `{image_path}`.")
            return

        orig_h, orig_w = bgr_image.shape[:2]
        img_key = f"{selected_folder}/{selected_image_name}"
        if img_key not in st.session_state["manual_corners"]:
            st.session_state["manual_corners"][img_key] = []

        # ----------------------------------------------------------------------
        # Step 1: Perspective Transformation (Auto Blue Table HSV or Manual 4-Corner Fallback)
        # ----------------------------------------------------------------------
        active_corners: np.ndarray | None = None
        corner_source_label = "Unwarped (Original)"
        hsv_mask: np.ndarray | None = None

        if enable_perspective:
            auto_corners, hsv_mask = detect_blue_table_corners(
                bgr_image,
                hsv_lower=(h_min, s_min, v_min),
                hsv_upper=(h_max, s_max, v_max),
                epsilon_factor=eps_factor,
            )

            manual_pts = st.session_state["manual_corners"][img_key]

            if not force_manual_corners and len(manual_pts) == 4:
                # If the user already completed 4 manual corners for this image, use them
                active_corners = order_corner_points(np.array(manual_pts, dtype=np.float32))
                corner_source_label = "Manual 4 Corners (800×800)"
            elif not force_manual_corners and auto_corners is not None:
                active_corners = auto_corners
                corner_source_label = "Auto HSV 4 Corners (800×800)"
            else:
                # Manual Fallback UI using streamlit-image-coordinates
                if len(manual_pts) == 4:
                    active_corners = order_corner_points(np.array(manual_pts, dtype=np.float32))
                    corner_source_label = "Manual 4 Corners (800×800)"
                else:
                    corner_source_label = f"Awaiting Manual Corners ({len(manual_pts)}/4)"

                st.warning(
                    "**Manual 4-Corner Perspective Fallback Active:** "
                    + (
                        "Automatic HSV contour detection did not find 4 corners of a blue table on this image. "
                        if auto_corners is None
                        else "Manual override is enabled. "
                    )
                    + f"Please click the **4 corners of the table** directly on the image below ({len(manual_pts)}/4 clicked)."
                )

                display_width = 750
                scale_factor = orig_w / float(display_width)
                overlay_bgr = draw_corner_overlay(bgr_image, manual_pts)
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

                click_col, ctrl_col = st.columns([3, 1])
                with click_col:
                    coords = streamlit_image_coordinates(
                        overlay_rgb,
                        width=display_width,
                        key=f"coord_clicker_{img_key}",
                    )
                    if coords is not None:
                        click_xy = (coords["x"], coords["y"])
                        if st.session_state["last_click_coord"].get(img_key) != click_xy:
                            st.session_state["last_click_coord"][img_key] = click_xy
                            if len(st.session_state["manual_corners"][img_key]) < 4:
                                real_x = float(coords["x"]) * scale_factor
                                real_y = float(coords["y"]) * scale_factor
                                st.session_state["manual_corners"][img_key].append((real_x, real_y))
                                st.rerun()

                with ctrl_col:
                    st.markdown("**Selected Corners (Original px):**")
                    for idx, (cx, cy) in enumerate(manual_pts):
                        st.write(f"{idx + 1}. `({int(round(cx))}, {int(round(cy))})`")
                    if st.button("Reset Clicked Corners", use_container_width=True):
                        st.session_state["manual_corners"][img_key] = []
                        st.session_state["last_click_coord"].pop(img_key, None)
                        st.rerun()
                    if auto_corners is not None and st.button("Use Auto HSV Corners", use_container_width=True):
                        st.session_state["manual_corners"][img_key] = []
                        st.rerun()

        # Route either the 800x800 perspective-warped bird's-eye image or the 800x800 cropped/resized image into HoughCircles
        if not enable_perspective:
            pipeline_input_bgr = resize_or_crop_to_square(bgr_image, target_size=WARP_SIZE)
            corner_source_label = f"Unwarped Resized/Cropped ({WARP_SIZE}×{WARP_SIZE})"
        elif active_corners is not None:
            pipeline_input_bgr = warp_table_perspective(bgr_image, active_corners, warp_size=WARP_SIZE)
        else:
            pipeline_input_bgr = resize_or_crop_to_square(bgr_image, target_size=WARP_SIZE)

        annotated_bgr, circle_count = detect_and_draw_circles(
            pipeline_input_bgr,
            min_dist=min_dist,
            param1=param1,
            param2=param2,
            min_radius=min_radius,
            max_radius=max_radius,
            blur_ksize=(blur_k, blur_k),
            blur_sigma=blur_sigma,
            reference_diameter_mm=reference_diameter_mm,
        )

        # Optional diagnostics expander to inspect the HSV mask and detected 4 corners on original image
        if enable_perspective and active_corners is not None:
            with st.expander("Inspect Detected Table Corners & Blue HSV Mask", expanded=False):
                diag_c1, diag_c2, diag_c3 = st.columns([2, 2, 1])
                with diag_c1:
                    corner_vis_bgr = draw_corner_overlay(bgr_image, active_corners)
                    st.image(
                        cv2.cvtColor(corner_vis_bgr, cv2.COLOR_BGR2RGB),
                        caption="4 Table Corners Ordered [TL, TR, BR, BL]",
                        use_container_width=True,
                    )
                with diag_c2:
                    if hsv_mask is not None:
                        st.image(hsv_mask, caption="Blue Table HSV Mask (cv2.inRange)", use_container_width=True)
                with diag_c3:
                    st.markdown("**Ordered Corners:**")
                    for lbl, pt in zip(["TL", "TR", "BR", "BL"], active_corners):
                        st.write(f"**{lbl}:** `({int(pt[0])}, {int(pt[1])})`")
                    if st.session_state["manual_corners"].get(img_key):
                        if st.button("Clear Manual Corners", key=f"clr_man_{img_key}", use_container_width=True):
                            st.session_state["manual_corners"][img_key] = []
                            st.session_state["last_click_coord"].pop(img_key, None)
                            st.rerun()

        # ----------------------------------------------------------------------
        # Step 2: Top Metrics & Side-by-Side Comparison
        # ----------------------------------------------------------------------
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Active Folder / Condition", f"{selected_folder} ({selected_condition})")
        m_col2.metric("Perspective Mode", corner_source_label)
        m_col3.metric("Detected Circles", circle_count)
        m_col4.metric("Pipeline Resolution", f"{pipeline_input_bgr.shape[1]} × {pipeline_input_bgr.shape[0]}")

        st.markdown("---")

        pipeline_input_rgb = cv2.cvtColor(pipeline_input_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_annot = st.columns(2)
        with col_orig:
            left_title = (
                "Flattened Bird's-Eye View (800×800)"
                if (enable_perspective and active_corners is not None)
                else "Resized / Cropped Raw Image (800×800)"
            )
            st.subheader(left_title)
            st.image(
                pipeline_input_rgb,
                caption=f"{selected_folder}/{selected_image_name} ({corner_source_label})",
                use_container_width=True,
            )

        with col_annot:
            st.subheader(f"Annotated Detection ({circle_count} circles)")
            st.image(
                annotated_rgb,
                caption=(
                    f"Condition: {selected_condition} | minDist={min_dist}, param1={param1}, "
                    f"param2={param2}, minRadius={min_radius}, maxRadius={max_radius}"
                ),
                use_container_width=True,
            )

        # ----------------------------------------------------------------------
        # Step 3: Evaluation Metrics Input & Session State Logging
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Manual Detection Evaluation & Result Logging")

        in_col1, in_col2, in_col3, btn_col = st.columns([1, 1, 1, 1])
        with in_col1:
            actual_correct = st.number_input(
                "Actual circles detected correctly",
                min_value=0,
                max_value=1000,
                value=int(circle_count),
                step=1,
            )
        with in_col2:
            false_positives = st.number_input(
                "False positives",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )
        with in_col3:
            missed_circles = st.number_input(
                "Missed circles",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )
        with btn_col:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Log Results", type="primary", use_container_width=True):
                log_entry = {
                    "Image Filename": selected_image_name,
                    "Folder": selected_folder,
                    "Condition": selected_condition,
                    "Perspective Mode": corner_source_label,
                    "minDist": min_dist,
                    "param1": param1,
                    "param2": param2,
                    "minRadius": min_radius,
                    "maxRadius": max_radius,
                    "Total Detected": circle_count,
                    "Actual circles detected correctly": actual_correct,
                    "False positives": false_positives,
                    "Missed circles": missed_circles,
                }
                st.session_state["results_log"].append(log_entry)
                st.toast(f"Logged results for {selected_image_name}!", icon="✅")

        # Exportable parameter snippet for detect_circles.py
        with st.expander("Copy Current Parameters for `detect_circles.py`"):
            st.code(
                f"MIN_DIST = {min_dist}\n"
                f"PARAM1 = {param1}\n"
                f"PARAM2 = {param2}\n"
                f"MIN_RADIUS = {min_radius}\n"
                f"MAX_RADIUS = {max_radius}\n"
                f"BLUR_KERNEL_SIZE = ({blur_k}, {blur_k})\n"
                f"BLUR_SIGMA = {blur_sigma}\n"
                f"HSV_LOWER_BLUE = ({h_min}, {s_min}, {v_min})\n"
                f"HSV_UPPER_BLUE = ({h_max}, {s_max}, {v_max})",
                language="python",
            )

        # ----------------------------------------------------------------------
        # Step 4: Running Log Table & CSV Export
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader(f"Session Results Log ({len(st.session_state['results_log'])} entries)")

        if st.session_state["results_log"]:
            log_df = pd.DataFrame(st.session_state["results_log"])
            st.dataframe(log_df, use_container_width=True)

            csv_bytes = log_df.to_csv(index=False).encode("utf-8")
            dl_col, clr_col, _ = st.columns([1, 1, 3])
            with dl_col:
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="hough_circle_tuning_log.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="dl_csv_tune_tab",
                )
            with clr_col:
                if st.button("Clear Session Log", use_container_width=True, key="clr_log_tune_tab"):
                    st.session_state["results_log"] = []
                    st.rerun()
        else:
            st.info(
                "No results logged in this session yet. Tune the parameters, enter the evaluation "
                "counts above, and click **Log Results** to add an entry to the table."
            )

    # --------------------------------------------------------------------------
    # Tab 2: Batch Processing & Review Mode
    # --------------------------------------------------------------------------
    with tab_batch:
        st.subheader("1. Batch Parameter Configuration (5 Iterations)")
        st.markdown(
            "Define **5 sets of Hough Circle parameters** below, choose the target folder, and click **Run Batch**. "
            "The app will loop through every image in that folder, apply each parameter set (with the active "
            "perspective transformation settings), and save all annotated images to `review_queue/` named with "
            "the parameter iteration number and original filename (e.g., `iter1_easy1.jpg`)."
        )

        b_ctrl1, b_ctrl2, b_ctrl3 = st.columns([2, 2, 2])
        with b_ctrl1:
            batch_folder_idx = folders.index(selected_folder) if selected_folder in folders else 0
            batch_folder = st.selectbox(
                "Target Folder for Batch Processing",
                options=folders,
                index=batch_folder_idx,
                key="batch_folder_select",
            )
        with b_ctrl2:
            batch_apply_perspective = st.toggle(
                "Apply Perspective Flattening",
                value=enable_perspective,
                key="batch_apply_perspective",
                help=(
                    "When enabled, warps the detected 4 table corners to 800×800. "
                    "When disabled, completely skips cv2.getPerspectiveTransform and cv2.warpPerspective, "
                    "safely center-cropping and resizing the raw image to 800×800."
                ),
            )
        with b_ctrl3:
            clear_existing_queue = st.checkbox(
                "Clear previous `review_queue` images before running batch",
                value=True,
                key="batch_clear_queue",
            )

        batch_condition = infer_default_condition(batch_folder, folder_labels)

        # Default 5 parameter variations around the base constants
        default_param_sets = [
            {"minDist": 25, "param1": 90, "param2": 22, "minRadius": 6, "maxRadius": 45},
            {"minDist": 30, "param1": 100, "param2": 28, "minRadius": 8, "maxRadius": 50},
            {"minDist": 35, "param1": 110, "param2": 32, "minRadius": 8, "maxRadius": 55},
            {"minDist": 40, "param1": 120, "param2": 36, "minRadius": 10, "maxRadius": 60},
            {"minDist": 45, "param1": 130, "param2": 40, "minRadius": 12, "maxRadius": 65},
        ]

        param_sets: list[dict[str, int]] = []
        set_cols = st.columns(5)
        for i in range(5):
            defaults = default_param_sets[i]
            with set_cols[i]:
                st.markdown(f"**Parameter Set #{i + 1}**")
                p_min_dist = st.number_input(
                    "minDist",
                    min_value=1,
                    max_value=500,
                    value=defaults["minDist"],
                    step=1,
                    key=f"b_mindist_{i + 1}",
                )
                p_param1 = st.number_input(
                    "param1",
                    min_value=10,
                    max_value=300,
                    value=defaults["param1"],
                    step=1,
                    key=f"b_param1_{i + 1}",
                )
                p_param2 = st.number_input(
                    "param2",
                    min_value=5,
                    max_value=150,
                    value=defaults["param2"],
                    step=1,
                    key=f"b_param2_{i + 1}",
                )
                p_min_rad = st.number_input(
                    "minRadius",
                    min_value=0,
                    max_value=300,
                    value=defaults["minRadius"],
                    step=1,
                    key=f"b_minrad_{i + 1}",
                )
                p_max_rad = st.number_input(
                    "maxRadius",
                    min_value=1,
                    max_value=500,
                    value=defaults["maxRadius"],
                    step=1,
                    key=f"b_maxrad_{i + 1}",
                )
                param_sets.append(
                    {
                        "iteration": i + 1,
                        "minDist": int(p_min_dist),
                        "param1": int(p_param1),
                        "param2": int(p_param2),
                        "minRadius": int(p_min_rad),
                        "maxRadius": int(p_max_rad),
                    }
                )

        if st.button("🚀 Run Batch", type="primary", use_container_width=True):
            target_dir = BASE_DIR / batch_folder
            batch_images = list_images_in_folder(target_dir)
            if not batch_images:
                st.error(f"No valid images found in `{batch_folder}/` to process.")
            else:
                REVIEW_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
                if clear_existing_queue:
                    for existing_file in REVIEW_QUEUE_DIR.iterdir():
                        if existing_file.is_file():
                            existing_file.unlink()
                    batch_meta: dict[str, dict] = {}
                else:
                    batch_meta = load_batch_metadata()

                total_tasks = len(batch_images) * len(param_sets)
                progress_bar = st.progress(0, text="Starting batch processing...")
                completed = 0

                for img_name in batch_images:
                    src_path = target_dir / img_name
                    raw_bgr = cv2.imread(str(src_path))
                    if raw_bgr is None:
                        continue

                    # Route through prepare_pipeline_image (warps if batch_apply_perspective=True, or safely crops/resizes to 800x800 if False)
                    img_key_b = f"{batch_folder}/{img_name}"
                    man_pts_b = st.session_state["manual_corners"].get(img_key_b, [])
                    proc_bgr, persp_label, _, _ = prepare_pipeline_image(
                        raw_bgr,
                        apply_perspective=batch_apply_perspective,
                        corners=np.array(man_pts_b, dtype=np.float32) if len(man_pts_b) == 4 else None,
                        hsv_lower=(h_min, s_min, v_min),
                        hsv_upper=(h_max, s_max, v_max),
                        epsilon_factor=eps_factor,
                        target_size=WARP_SIZE,
                    )

                    for pset in param_sets:
                        it_num = pset["iteration"]
                        annot_bgr, det_count = detect_and_draw_circles(
                            proc_bgr,
                            min_dist=pset["minDist"],
                            param1=pset["param1"],
                            param2=pset["param2"],
                            min_radius=pset["minRadius"],
                            max_radius=pset["maxRadius"],
                            blur_ksize=(blur_k, blur_k),
                            blur_sigma=blur_sigma,
                            reference_diameter_mm=reference_diameter_mm,
                        )
                        out_filename = f"iter{it_num}_{img_name}"
                        out_path = REVIEW_QUEUE_DIR / out_filename
                        cv2.imwrite(str(out_path), annot_bgr)

                        batch_meta[out_filename] = {
                            "Review Filename": out_filename,
                            "Original Filename": img_name,
                            "Folder": batch_folder,
                            "Condition": batch_condition,
                            "Parameter Iteration": it_num,
                            "Perspective Mode": persp_label,
                            "minDist": pset["minDist"],
                            "param1": pset["param1"],
                            "param2": pset["param2"],
                            "minRadius": pset["minRadius"],
                            "maxRadius": pset["maxRadius"],
                            "Total Detected": int(det_count),
                        }

                        completed += 1
                        progress_bar.progress(
                            completed / total_tasks,
                            text=f"Processed {completed}/{total_tasks}: `{out_filename}` ({det_count} circles)",
                        )

                save_batch_metadata(batch_meta)
                st.session_state["review_index"] = 0
                st.success(
                    f"Batch complete! Saved {completed} annotated images to `review_queue/`. "
                    "Use **Review Mode** below to inspect and log results one by one."
                )

        # ----------------------------------------------------------------------
        # Section 2: Review Mode UI (One-by-One Annotated Image Review)
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Review Mode (`review_queue` Folder)")

        REVIEW_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        review_files = list_images_in_folder(REVIEW_QUEUE_DIR)
        batch_meta_all = load_batch_metadata()

        if not review_files:
            st.info(
                "The `review_queue/` folder is currently empty. Configure your 5 parameter sets above "
                "and click **🚀 Run Batch** to populate the review queue."
            )
        else:
            # Clamp review_index to valid range
            if st.session_state["review_index"] >= len(review_files):
                st.session_state["review_index"] = len(review_files) - 1
            if st.session_state["review_index"] < 0:
                st.session_state["review_index"] = 0

            r_idx = st.session_state["review_index"]
            current_review_name = review_files[r_idx]
            current_review_path = REVIEW_QUEUE_DIR / current_review_name

            # Navigation Bar
            nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([1, 2, 1, 2])
            with nav_c1:
                if st.button("◀ Prev Image", use_container_width=True, disabled=(r_idx == 0), key="rev_prev"):
                    st.session_state["review_index"] = r_idx - 1
                    st.rerun()
            with nav_c2:
                jumped_file = st.selectbox(
                    f"Review Queue Item ({r_idx + 1} of {len(review_files)})",
                    options=review_files,
                    index=r_idx,
                    key="rev_jump_select",
                )
                if jumped_file != current_review_name:
                    st.session_state["review_index"] = review_files.index(jumped_file)
                    st.rerun()
            with nav_c3:
                if st.button(
                    "Next Image ▶",
                    use_container_width=True,
                    disabled=(r_idx == len(review_files) - 1),
                    key="rev_next",
                ):
                    st.session_state["review_index"] = r_idx + 1
                    st.rerun()
            with nav_c4:
                logged_review_files = {
                    row.get("Review Filename") for row in st.session_state["results_log"] if "Review Filename" in row
                }
                is_already_logged = current_review_name in logged_review_files
                st.metric(
                    "Queue Progress",
                    f"{r_idx + 1} / {len(review_files)}",
                    delta="Logged ✓" if is_already_logged else "Unlogged",
                )

            # Parse metadata for current review image (fallback to filename parsing if metadata missing)
            item_meta = batch_meta_all.get(current_review_name, {})
            orig_fname = item_meta.get("Original Filename", current_review_name)
            item_folder = item_meta.get("Folder", batch_folder)
            item_cond = item_meta.get("Condition", infer_default_condition(item_folder, folder_labels))
            item_iter = item_meta.get("Parameter Iteration", "N/A")
            item_persp = item_meta.get("Perspective Mode", "Saved Batch Image")
            item_mindist = item_meta.get("minDist", MIN_DIST)
            item_p1 = item_meta.get("param1", PARAM1)
            item_p2 = item_meta.get("param2", PARAM2)
            item_minr = item_meta.get("minRadius", MIN_RADIUS)
            item_maxr = item_meta.get("maxRadius", MAX_RADIUS)
            item_detected = int(item_meta.get("Total Detected", 0))

            rev_img_col, rev_form_col = st.columns([3, 2])
            with rev_img_col:
                rev_bgr = cv2.imread(str(current_review_path))
                if rev_bgr is not None:
                    rev_rgb = cv2.cvtColor(rev_bgr, cv2.COLOR_BGR2RGB)
                    st.image(
                        rev_rgb,
                        caption=(
                            f"{current_review_name} | Iter #{item_iter} | Orig: {orig_fname} "
                            f"({item_detected} circles detected)"
                        ),
                        use_container_width=True,
                    )
                else:
                    st.error(f"Could not read image `{current_review_path}`.")

            with rev_form_col:
                st.markdown(f"### Inspecting: `{current_review_name}`")
                st.markdown(
                    f"- **Original File:** `{orig_fname}` (`{item_folder}` — **{item_cond}**)\n"
                    f"- **Parameter Iteration:** `Set #{item_iter}`\n"
                    f"- **Hough Parameters:** `minDist={item_mindist}`, `param1={item_p1}`, "
                    f"`param2={item_p2}`, `minRadius={item_minr}`, `maxRadius={item_maxr}`\n"
                    f"- **Total Circles Detected:** `{item_detected}`"
                )
                st.markdown("---")
                rev_fp = st.number_input(
                    "False Positives",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    step=1,
                    key=f"rev_fp_{current_review_name}",
                )
                rev_missed = st.number_input(
                    "Missed Circles",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    step=1,
                    key=f"rev_missed_{current_review_name}",
                )
                inferred_correct = max(0, item_detected - int(rev_fp))
                rev_correct = st.number_input(
                    "Actual circles detected correctly",
                    min_value=0,
                    max_value=1000,
                    value=inferred_correct,
                    step=1,
                    key=f"rev_corr_{current_review_name}",
                    help="Automatically defaults to (Total Detected - False Positives), or adjust manually.",
                )

                auto_advance = st.checkbox(
                    "Auto-advance to next image after logging",
                    value=True,
                    key="rev_auto_advance",
                )

                if st.button(
                    "Log Review Metrics to Session DataFrame",
                    type="primary",
                    use_container_width=True,
                    key=f"btn_log_rev_{current_review_name}",
                ):
                    rev_entry = {
                        "Review Filename": current_review_name,
                        "Image Filename": orig_fname,
                        "Folder": item_folder,
                        "Condition": item_cond,
                        "Parameter Iteration": item_iter,
                        "Perspective Mode": item_persp,
                        "minDist": item_mindist,
                        "param1": item_p1,
                        "param2": item_p2,
                        "minRadius": item_minr,
                        "maxRadius": item_maxr,
                        "Total Detected": item_detected,
                        "Actual circles detected correctly": int(rev_correct),
                        "False positives": int(rev_fp),
                        "Missed circles": int(rev_missed),
                    }
                    st.session_state["results_log"].append(rev_entry)
                    st.toast(f"Logged `{current_review_name}` to Session DataFrame!", icon="✅")
                    if auto_advance and r_idx < len(review_files) - 1:
                        st.session_state["review_index"] = r_idx + 1
                        st.rerun()

        # ----------------------------------------------------------------------
        # Section 3: Session DataFrame & Final CSV Export in Batch/Review Tab
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader(f"3. Logged Evaluation DataFrame ({len(st.session_state['results_log'])} entries)")

        if st.session_state["results_log"]:
            batch_log_df = pd.DataFrame(st.session_state["results_log"])
            st.dataframe(batch_log_df, use_container_width=True)

            batch_csv_bytes = batch_log_df.to_csv(index=False).encode("utf-8")
            b_dl_col, b_clr_col, _ = st.columns([1, 1, 3])
            with b_dl_col:
                st.download_button(
                    label="📥 Export Final CSV",
                    data=batch_csv_bytes,
                    file_name="hough_circle_batch_review_log.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="dl_csv_batch_tab",
                )
            with b_clr_col:
                if st.button("Clear Session DataFrame", use_container_width=True, key="clr_log_batch_tab"):
                    st.session_state["results_log"] = []
                    st.rerun()
        else:
            st.info("No review entries logged yet. Inspect images in **Review Mode** above and click **Log Review Metrics**.")


if __name__ == "__main__":
    main()
