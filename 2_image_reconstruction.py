import argparse
import os
import re
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))


def group_images_by_prefix_recursive(root_dir):
    pattern_split = re.compile(r"(.+)_split_(\d+)\.(\w+)$")
    pattern_grid = re.compile(r"(.+)_grid(\d+)x(\d+)_(\d+)_(\d+)\.(\w+)$")
    pattern_longedge = re.compile(r"(.+)_longedge(\d+)_(\d+)\.(\w+)$")
    pattern_simple_grid = re.compile(r"(.+)_([0-9]+)_([0-9]+)\.(\w+)$")

    split_groups = defaultdict(list)
    grid_groups = defaultdict(list)
    longedge_groups = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not is_image_file(file):
                continue

            abs_path = os.path.join(dirpath, file)
            rel_path = os.path.relpath(abs_path, root_dir)
            rel_dir = os.path.dirname(rel_path)

            match_split = pattern_split.match(file)
            match_grid = pattern_grid.match(file)
            match_longedge = pattern_longedge.match(file)
            match_simple_grid = pattern_simple_grid.match(file)

            if match_grid:
                base = match_grid.group(1)
                grid_w = int(match_grid.group(2))
                grid_h = int(match_grid.group(3))
                row = int(match_grid.group(4))
                col = int(match_grid.group(5))
                key = (rel_dir, base, f"grid{grid_w}x{grid_h}")
                grid_groups[key].append((row, col, abs_path))
            elif match_longedge:
                base = match_longedge.group(1)
                seg = int(match_longedge.group(2))
                idx = int(match_longedge.group(3))
                key = (rel_dir, base, f"longedge{seg}")
                longedge_groups[key].append((idx, abs_path))
            elif match_simple_grid:
                base = match_simple_grid.group(1)
                row = int(match_simple_grid.group(2))
                col = int(match_simple_grid.group(3))
                key = (rel_dir, base, "simple_grid")
                grid_groups[key].append((row, col, abs_path))
            elif match_split:
                base = match_split.group(1)
                idx = int(match_split.group(2))
                key = (rel_dir, base, "split")
                split_groups[key].append((idx, abs_path))

    return split_groups, grid_groups, longedge_groups


def merge_images_horizontally(image_paths, save_path):
    images = [Image.open(p) for _, p in sorted(image_paths)]
    mode = images[0].mode
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images)

    new_img = Image.new(mode, (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    new_img.save(save_path)


def merge_images_grid(image_tuples, save_path):
    sorted_images = sorted(image_tuples, key=lambda x: (x[0], x[1]))
    rows = max(r for r, c, _ in sorted_images) + 1
    cols = max(c for r, c, _ in sorted_images) + 1

    img_grid = [[None] * cols for _ in range(rows)]
    for row, col, path in sorted_images:
        img_grid[row][col] = Image.open(path)

    mode = img_grid[0][0].mode
    col_widths = [
        max(img_grid[row][col].width for row in range(rows) if img_grid[row][col] is not None)
        for col in range(cols)
    ]
    row_heights = [
        max(img_grid[row][col].height for col in range(cols) if img_grid[row][col] is not None)
        for row in range(rows)
    ]
    x_offsets = [0]
    y_offsets = [0]
    for width in col_widths[:-1]:
        x_offsets.append(x_offsets[-1] + width)
    for height in row_heights[:-1]:
        y_offsets.append(y_offsets[-1] + height)

    new_img = Image.new(mode, (sum(col_widths), sum(row_heights)))

    for row in range(rows):
        for col in range(cols):
            if img_grid[row][col]:
                new_img.paste(img_grid[row][col], (x_offsets[col], y_offsets[row]))

    new_img.save(save_path)


def reconstruct_images_recursive(input_dir, output_dir):
    split_groups, grid_groups, longedge_groups = group_images_by_prefix_recursive(input_dir)
    total = len(split_groups) + len(grid_groups) + len(longedge_groups)

    pbar = tqdm(total=total, desc=f"Reconstructing {os.path.basename(input_dir)}", ncols=100)

    for (rel_dir, base, _), images in split_groups.items():
        ext = os.path.splitext(images[0][1])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        merge_images_horizontally(images, os.path.join(output_subdir, f"{base}{ext}"))
        pbar.update(1)

    for (rel_dir, base, _), images in grid_groups.items():
        ext = os.path.splitext(images[0][2])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        merge_images_grid(images, os.path.join(output_subdir, f"{base}{ext}"))
        pbar.update(1)

    for (rel_dir, base, _), images in longedge_groups.items():
        ext = os.path.splitext(images[0][1])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        images_sorted = sorted(images, key=lambda x: x[0])
        merge_images_horizontally(images_sorted, os.path.join(output_subdir, f"{base}{ext}"))
        pbar.update(1)

    pbar.close()
    print(f"Reconstruction finished: {input_dir} -> {output_dir}")


def main(results_root):
    categories = [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    ]

    for category in categories:
        input_images = os.path.join(results_root, "anomaly_images", category)
        output_images = os.path.join(results_root, "anomaly_images", f"{category}_merge")

        if not os.path.isdir(input_images):
            print(f"Skipping {category}: missing input directory {input_images}")
            continue

        reconstruct_images_recursive(input_images, output_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        nargs="?",
        default="./results",
        help="Root results directory containing anomaly_images.",
    )
    args = parser.parse_args()
    main(args.results_root)
