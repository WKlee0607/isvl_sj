import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


SRC_ROOT = './mvtec_ad_2'
DST_ROOT = './mvtec_ad_2_aug'
CATEGORIES = [
    'can',
    'fabric',
    'fruit_jelly',
    'rice',
    'sheet_metal',
    'vial',
    'wallplugs',
    'walnuts',
]
EVAL_SUBSETS = ['test_private', 'test_private_mixed', 'test_public', 'validation']
COMMON_GRID_SIZE = (4, 4)


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))


def is_original_image(filename):
    name, _ = os.path.splitext(filename)
    if '_split_' in name or 'longedge' in name or 'grid' in name:
        return False

    parts = name.split('_')
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return False
    return True


def collect_image_paths(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file) and is_original_image(file):
                image_paths.append(os.path.join(root, file))
    return image_paths


def save_sub_image(sub_img, save_dir, base_name, suffix, save_format, mode_tag=None):
    if mode_tag:
        save_path = os.path.join(save_dir, f"{base_name}_{mode_tag}_{suffix}.{save_format.lower()}")
    else:
        save_path = os.path.join(save_dir, f"{base_name}_{suffix}.{save_format.lower()}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return

    if save_format.lower() in ('jpg', 'jpeg'):
        sub_img = sub_img.convert('RGB')
    sub_img.save(save_path)


def split_image_grid(img, base_name, save_dir, grid_size, save_format):
    img_width, img_height = img.size
    grid_w, grid_h = grid_size
    sub_img_width = img_width // grid_w
    sub_img_height = img_height // grid_h
    mode_tag = f"grid{grid_w}x{grid_h}"

    for row in range(grid_h):
        for col in range(grid_w):
            left = col * sub_img_width
            upper = row * sub_img_height
            right = (col + 1) * sub_img_width if col != grid_w - 1 else img_width
            lower = (row + 1) * sub_img_height if row != grid_h - 1 else img_height
            sub_img = img.crop((left, upper, right, lower))
            save_sub_image(sub_img, save_dir, base_name, f"{row}_{col}", save_format, mode_tag=mode_tag)


def split_image(
    image_path,
    save_dir,
    grid_size=COMMON_GRID_SIZE,
    save_format='png',
    input_root_dir=None,
):
    with Image.open(image_path) as img:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if input_root_dir:
            relative_path = os.path.relpath(image_path, start=input_root_dir)
            relative_dir = os.path.dirname(relative_path)
            save_dir = os.path.join(save_dir, relative_dir)

        split_image_grid(img, base_name, save_dir, grid_size, save_format)


def process_dataset(input_dir, output_dir, grid_size=COMMON_GRID_SIZE, save_format='png', max_workers=8):
    if not os.path.isdir(input_dir):
        print(f"Skipping split: missing source directory {input_dir}")
        return

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print(f"No original images found in {input_dir}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                split_image,
                image_path,
                output_dir,
                grid_size,
                save_format,
                input_dir,
            )
            for image_path in image_paths
        ]

        pbar = tqdm(total=len(image_paths), desc=f"Splitting {input_dir}", ncols=100)
        for future in as_completed(futures):
            future.result()
            pbar.update(1)
        pbar.close()


def process_category(category):
    print(f"Processing {category}")
    src_category_dir = os.path.join(SRC_ROOT, category)
    dst_category_dir = os.path.join(DST_ROOT, category)

    if not os.path.isdir(src_category_dir):
        print(f"Skipping {category}: missing source directory {src_category_dir}")
        return

    if os.path.exists(dst_category_dir):
        shutil.rmtree(dst_category_dir)

    train_dir = os.path.join(src_category_dir, 'train')
    train_output = os.path.join(dst_category_dir, 'train')

    process_dataset(train_dir, train_output, grid_size=COMMON_GRID_SIZE)

    for subset in EVAL_SUBSETS:
        subset_dir = os.path.join(src_category_dir, subset)
        subset_output = os.path.join(dst_category_dir, subset)
        process_dataset(subset_dir, subset_output, grid_size=COMMON_GRID_SIZE)


if __name__ == "__main__":
    for category in CATEGORIES:
        process_category(category)
