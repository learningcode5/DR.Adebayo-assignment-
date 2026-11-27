"""
Standalone script to sequentially rename image files inside the 'easy', 'moderate',
and 'hard' folders in the local project directory (e.g., easy1.jpg, easy2.jpg, ...,
hard1.jpg, hard2.jpg).

Features:
- Retains the original file extension (case-normalized to lowercase, e.g., .jpg, .png).
- Handles existing files safely using a two-pass rename strategy (via temporary UUID names)
  so re-running the script when files like 'hard1.jpg' already exist never causes collisions
  or overwrites.
"""

from pathlib import Path
import re
import uuid

BASE_DIR = Path(__file__).resolve().parent
TARGET_FOLDERS = ["easy", "moderate", "hard"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(path: Path) -> tuple:
    """Sort key so already-numbered files (e.g. hard2.jpg, hard10.jpg) stay in numerical order."""
    parts = re.split(r"(\d+)", path.stem.lower())
    converted = [int(p) if p.isdigit() else p for p in parts]
    return (converted, path.name.lower())


def rename_images_in_folder(folder_path: Path) -> list[tuple[str, str]]:
    """
    Sequentially rename all valid image files in `folder_path` to `<folder_name><index><ext>`.
    Uses a safe two-pass rename to avoid overwriting existing sequentially named files.
    """
    if not folder_path.is_dir():
        print(f"[SKIP] Folder not found: {folder_path}")
        return []

    prefix = folder_path.name.lower()
    image_files = sorted(
        [
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS
        ],
        key=natural_sort_key,
    )

    if not image_files:
        print(f"[INFO] No image files found in '{folder_path.name}/'.")
        return []

    # Pass 1: Rename all files to unique temporary filenames to prevent any naming collisions
    temp_records: list[tuple[str, Path, str]] = []
    for img_path in image_files:
        orig_name = img_path.name
        orig_ext = img_path.suffix.lower()  # Retain original extension
        temp_name = f"__tmp_rename_{uuid.uuid4().hex}{orig_ext}"
        temp_path = folder_path / temp_name
        img_path.rename(temp_path)
        temp_records.append((orig_name, temp_path, orig_ext))

    # Pass 2: Rename from temporary filenames to final sequential names (e.g., easy1.jpg, easy2.jpg)
    renamed_pairs: list[tuple[str, str]] = []
    for idx, (orig_name, temp_path, orig_ext) in enumerate(temp_records, start=1):
        new_name = f"{prefix}{idx}{orig_ext}"
        final_path = folder_path / new_name

        # Extra safety check in case an unexpected non-image or locked file shares the name
        counter = idx
        while final_path.exists():
            counter += 1
            new_name = f"{prefix}{counter}{orig_ext}"
            final_path = folder_path / new_name

        temp_path.rename(final_path)
        renamed_pairs.append((orig_name, new_name))

    return renamed_pairs


def main() -> None:
    print(f"Scanning project directory: {BASE_DIR}\n")
    for folder_name in TARGET_FOLDERS:
        folder_path = BASE_DIR / folder_name
        renamed = rename_images_in_folder(folder_path)
        if renamed:
            print(f"Renamed {len(renamed)} image(s) in '{folder_name}/':")
            for old_name, new_name in renamed:
                print(f"  {old_name}  -->  {new_name}")
            print()


if __name__ == "__main__":
    main()
