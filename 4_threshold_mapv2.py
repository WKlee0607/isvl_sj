import argparse
import os

import cv2
import numpy as np
import tifffile

# DINOv3
# thresholds = {
#     "can": {
#         "test_private": 39, #55, 
#         "test_private_mixed": 39, #55 
#     },
#     "fabric": {
#         "test_private": 23, #69,
#         "test_private_mixed": 23 #69,
#     },
#     "fruit_jelly": { # 95로바꿔
#         "test_private": 39, #86,
#         "test_private_mixed": 39 #86
#     },
#     "rice": {
#         "test_private": 19, #54,
#         "test_private_mixed": 19 #54
#     },
#     "sheet_metal": {
#         "test_private": 27,#63,
#         "test_private_mixed": 27 #65
#     },
#     "vial": { # 95로바꿔
#         "test_private": 42, #79,
#         "test_private_mixed": 42 #79
#     },
#     "wallplugs": {
#         "test_private": 48, #99,
#         "test_private_mixed": 48 #108
#     },
#     "walnuts": {
#         "test_private": 23, #67,
#         "test_private_mixed": 23
#     }
# }

# Epoch 10
# thresholds = {
#     "can": {
#         "test_private": 42,  
#         "test_private_mixed": 42, 
#     },
#     "fabric": {
#         "test_private": 18, 
#         "test_private_mixed": 18 
#     },
#     "fruit_jelly": { # 95로바꿔
#         "test_private": 30, 
#         "test_private_mixed": 30 
#     },
#     "rice": {
#         "test_private": 22, 
#         "test_private_mixed": 22 
#     },
#     "sheet_metal": {
#         "test_private": 25,
#         "test_private_mixed": 25 
#     },
#     "vial": { # 95로바꿔
#         "test_private": 28, 
#         "test_private_mixed": 28 
#     },
#     "wallplugs": {
#         "test_private": 33, 
#         "test_private_mixed": 33 
#     },
#     "walnuts": {
#         "test_private": 25, 
#         "test_private_mixed": 25
#     }
# }


# Epoch 10 test
thresholds = {
    "can": {
        "test_private": 42,  
        "test_private_mixed": 52, #47, #42, 
    },
    "fabric": {
        "test_private": 13, #23, #18, 
        "test_private_mixed":13, #23 #18 
    },
    "fruit_jelly": { # 95로바꿔
        "test_private": 38, #35, #30, 
        "test_private_mixed": 38, #35, #30 
    },
    "rice": {
        "test_private": 22, 
        "test_private_mixed": 22 
    },
    "sheet_metal": {
        "test_private": 25,
        "test_private_mixed": 25 
    },
    "vial": { # 95로바꿔
        "test_private": 28, 
        "test_private_mixed": 28 
    },
    "wallplugs": {
        "test_private": 33, 
        "test_private_mixed": 33 
    },
    "walnuts": {
        "test_private": 25, 
        "test_private_mixed": 25
    }
}




# Epoch 15 -> 20도 그냥 이걸로 ㄱ
# thresholds = {
#     "can": {"test_private": 39, "test_private_mixed": 39},
#     "fabric": {"test_private": 17, "test_private_mixed": 17},
#     "fruit_jelly": {"test_private": 24, "test_private_mixed": 24},
#     "rice": {"test_private": 18, "test_private_mixed": 18},
#     "sheet_metal": {"test_private": 22, "test_private_mixed": 22},
#     "vial": {"test_private": 25, "test_private_mixed": 25},
#     "wallplugs": {"test_private": 30, "test_private_mixed": 30},
#     "walnuts": {"test_private": 19, "test_private_mixed": 19},
# }



USE_HYBRID_ADAPTIVE_THRESHOLD = False

# adaptive = 0.5 * public_threshold + 0.3 * percentile + 0.2 * mean/std
HYBRID_PERCENTILE = 99.3
HYBRID_STD_SCALE = 2.5
HYBRID_BASE_WEIGHT = 0.5
HYBRID_PERCENTILE_WEIGHT = 0.3
HYBRID_STAT_WEIGHT = 0.2

# Keep per-image thresholds from drifting too far from the public optimum.
HYBRID_MIN_RATIO = 0.7
HYBRID_MAX_RATIO = 1.5

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_anomaly_image(file_path):
    try:
        if file_path.lower().endswith((".tif", ".tiff")):
            img = tifffile.imread(file_path)
            img = np.asarray(img)
            if img.ndim == 3:
                img = img[..., 0]
            return np.clip(img, 0, 255).astype(np.uint8)

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img
    except Exception as exc:
        print(f"Failed to read image: {file_path} ({exc})")
        return None


def get_hybrid_adaptive_threshold(img, base_thresh):
    pixels = img.reshape(-1).astype(np.float32)
    percentile_thresh = np.percentile(pixels, HYBRID_PERCENTILE)
    stat_thresh = float(np.mean(pixels) + HYBRID_STD_SCALE * np.std(pixels))

    adaptive_thresh = (
        HYBRID_BASE_WEIGHT * base_thresh
        + HYBRID_PERCENTILE_WEIGHT * percentile_thresh
        + HYBRID_STAT_WEIGHT * stat_thresh
    )

    lower = base_thresh * HYBRID_MIN_RATIO
    upper = base_thresh * HYBRID_MAX_RATIO
    adaptive_thresh = np.clip(adaptive_thresh, lower, upper)
    return int(np.clip(round(adaptive_thresh), 0, 255))


def get_base_threshold(category, subfolder):
    if category not in thresholds:
        return None

    category_thresholds = thresholds[category]
    if subfolder in category_thresholds:
        return category_thresholds[subfolder]
    return category_thresholds.get("default")


def threshold_and_save_images_recursive(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    saved_count = 0
    skipped_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTS):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, input_dir)
            rel_parts = rel_path.split(os.sep)

            if len(rel_parts) < 2:
                print(f"Warning: File {file_path} not in expected category/subfolder structure.")
                skipped_count += 1
                continue

            category = rel_parts[0]
            subfolder = rel_parts[1]
            base_thresh = get_base_threshold(category, subfolder)
            if base_thresh is None:
                print(f"Warning: No threshold found for {category}/{subfolder}, skipping {file_path}")
                skipped_count += 1
                continue

            img = read_anomaly_image(file_path)
            if img is None:
                failed_files.append(file_path)
                continue

            thresh = base_thresh
            if USE_HYBRID_ADAPTIVE_THRESHOLD:
                thresh = get_hybrid_adaptive_threshold(img, base_thresh)

            _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

            base_filename = os.path.splitext(file)[0] + ".png"
            save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            save_path = os.path.join(save_dir, base_filename)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(save_path, binary_img)
            saved_count += 1

    if failed_files:
        print("\nFailed files:")
        for file_path in failed_files:
            print(f"  {file_path}")
        raise RuntimeError(f"Failed to read {len(failed_files)} image(s).")

    print(f"Saved {saved_count} thresholded images. Skipped {skipped_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        nargs="?",
        default="./results",
        help="Root results directory containing anomaly_images.",
    )
    args = parser.parse_args()

    input_folder = os.path.join(args.results_root, "anomaly_images")
    output_folder = os.path.join(args.results_root, "anomaly_images_thresholded")
    threshold_and_save_images_recursive(input_folder, output_folder)
