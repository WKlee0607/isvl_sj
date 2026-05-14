import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MIN_COMPONENT_AREA_RATIO = 0.00002
MIN_COMPONENT_AREA_PIXELS = 8


def remove_tiny_components(binary_img):
    binary_img = (binary_img > 127).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    min_area = max(MIN_COMPONENT_AREA_PIXELS, int(binary_img.shape[0] * binary_img.shape[1] * MIN_COMPONENT_AREA_RATIO))
    cleaned = np.zeros_like(binary_img, dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def process_thresholded_images(root_dir):
    if not os.path.isdir(root_dir):
        print(f'Skipping post-processing: missing directory {root_dir}')
        return

    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.lower().endswith(IMAGE_EXTS)
    ]

    for image_path in tqdm(image_paths, desc='Uniform post-processing', ncols=100):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Warning: failed to read {image_path}')
            continue

        cleaned = remove_tiny_components(img)
        cv2.imwrite(image_path, cleaned)

    print(f'Uniform post-processing complete. Processed {len(image_paths)} image(s).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_root',
        nargs='?',
        default='./results',
        help='Root results directory containing anomaly_images_thresholded.',
    )
    args = parser.parse_args()
    process_thresholded_images(os.path.join(args.results_root, 'anomaly_images_thresholded'))
