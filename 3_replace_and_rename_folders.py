import argparse
import os
import shutil


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


def replace_with_merged_outputs(root_dir, categories=CATEGORIES):
    for category in categories:
        original_path = os.path.join(root_dir, category)
        merged_path = os.path.join(root_dir, f'{category}_merge')

        if not os.path.isdir(merged_path):
            print(f'Skipping {category}: no merged directory at {merged_path}')
            continue

        if os.path.exists(original_path):
            shutil.rmtree(original_path)
            print(f'Removed original split directory: {original_path}')

        os.rename(merged_path, original_path)
        print(f'Renamed merged directory: {merged_path} -> {original_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_root',
        nargs='?',
        default='./results',
        help='Root results directory containing anomaly_images.',
    )
    args = parser.parse_args()
    replace_with_merged_outputs(os.path.join(args.results_root, 'anomaly_images'))
