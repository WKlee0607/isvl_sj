# VAND 4.0 Industrial Track

This repository contains an INP-Former based anomaly detection pipeline for the
VAND Industrial Track regular setting. The pipeline trains the same model
architecture and uses the same preprocessing, inference, thresholding, and
postprocessing code for all object categories.

## Method Summary

The model uses a DINOv3 ViT-L/16 backbone as a frozen feature extractor and
trains INP-Former decoder/prototype modules on normal training images. During
inference, anomaly scores are computed from the cosine distance between encoder
features and decoder reconstruction features. The resulting anomaly maps are
saved as TIFF files and postprocessed into the required submission structure.

Categories:

```text
can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
```

## Environment

Experiments were run with one NVIDIA RTX 3090 GPU with 24 GB memory.

```bash
conda create -n cvprw python=3.10
conda activate cvprw
pip install -r requirements.txt
```

Important package versions are listed in `requirements.txt`, including:

```text
torch==2.7.1
torchvision==0.22.1
timm==0.9.12
opencv-python==4.11.0.86
tifffile==2023.7.10
ADEval==1.1.0
```

## Pretrained Backbone

The default backbone is DINOv3 ViT-L/16.

Download the pretrained weight and place it at:

```text
./pre_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Weight URL:

```text
TODO: add public pretrained weight link here
```

DINOv3 repository URL:

```text
https://github.com/facebookresearch/dinov3
```

Please also check and follow the license terms of the pretrained backbone and
weights.

The run script sets:

```bash
export DINOV3_REPO="${DINOV3_REPO:-./dinov3}"
export DINOV3_WEIGHTS="./pre_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
```

## Dataset Preparation

Unzip the MVTec AD 2 dataset into:

```text
./mvtec_ad_2
```

Expected structure:

```text
./mvtec_ad_2/
  can/
  fabric/
  fruit_jelly/
  rice/
  sheet_metal/
  vial/
  wallplugs/
  walnuts/
```

Run the splitter to create the preprocessed dataset:

```bash
python 1_image_splitter.py
```

This creates:

```text
./mvtec_ad_2_aug
```

The splitter uses the same grid-splitting procedure for all categories.

## Training

The training entry point is:

```bash
bash run_vand.sh
```

Main training settings in `run_vand.sh`:

```bash
DATA_PATH="./mvtec_ad_2_aug"
TOTAL_EPOCHS=15
RESULTS_DIR="./results"
```

Each category is trained with the same architecture and hyperparameters. The
category name only selects the corresponding normal training data directory.

Checkpoints are saved under `--save_dir`. The current default in `isvl.py` is:

```text
./saved_results
```

Make sure `--save_dir` and `TOTAL_EPOCHS` are consistent with the checkpoint
directory used for inference.

The checkpoint naming pattern is:

```text
{save_dir}/
  INP-Former-Multi-Class_dataset=./mvtec_ad_2_Encoder=dinov3_vitl16_Resize=448_Crop=392_INP_num=6_CLASS={category}/
    model.pth
```

## Inference

Inference uses:

```bash
python isvl.py \
  --data_path "$DATA_PATH" \
  --item_list can \
  --total_epochs "$TOTAL_EPOCHS" \
  --phase val \
  --results_dir "$RESULTS_DIR"
```

`--results_dir` controls where anomaly maps are written. The output structure is:

```text
{RESULTS_DIR}/anomaly_images/{category}/...
```

The same command pattern is repeated for every category.

## Thresholding

Thresholding is performed by:

```bash
python 4_threshold_mapv2.py "$RESULTS_DIR"
```

The script reads:

```text
{RESULTS_DIR}/anomaly_images
```

and writes:

```text
{RESULTS_DIR}/anomaly_images_thresholded
```

Current threshold configuration is stored in `4_threshold_mapv2.py`.

Threshold source/configuration:

Thresholds are computed with `compute_validation_thresholds.py` from anomaly
scores on the `validation/good` split. The script loads the trained checkpoints,
runs inference on normal validation images, and applies a fixed rule:

```text
threshold = mean(validation_good_scores) + k * std(validation_good_scores)
```

Example command:

```bash
python compute_validation_thresholds.py \
  --data_path "./mvtec_ad_2_aug" \
  --save_dir "./saved_results" \
  --k 3.0 \
  --threshold_scope global \
  --output "./validation_thresholds.json"
```

The resulting thresholds are written to `validation_thresholds.json`. The final
threshold values used for submission were then entered into `4_threshold_mapv2.py`
before running the final postprocessing step.

This threshold procedure does not use `test_public`, `test_private`, or
`test_private_mixed` data for threshold optimization.

## Postprocessing And Submission

The postprocessing commands are:

```bash
python 2_image_reconstruction.py "$RESULTS_DIR"
python 3_replace_and_rename_folders.py "$RESULTS_DIR"
python 4_threshold_mapv2.py "$RESULTS_DIR"
python 5_post_image_process.py "$RESULTS_DIR"
python 7_convert_tiff_to_float16.py "$RESULTS_DIR"
python 8_check_and_prepare_data_for_upload.py "$RESULTS_DIR"
```

Step summary:

```text
2_image_reconstruction.py
  Reconstructs grid/split anomaly maps back to original image layout.

3_replace_and_rename_folders.py
  Replaces split category folders with reconstructed category folders.

4_threshold_mapv2.py
  Converts continuous anomaly maps into binary thresholded masks.

5_post_image_process.py
  Removes tiny connected components from thresholded masks using one uniform rule.

7_convert_tiff_to_float16.py
  Converts continuous anomaly TIFF files to float16 TIFF format.

8_check_and_prepare_data_for_upload.py
  Checks the submission directory and creates the final archive.
```

The final archive is written as:

```text
results.tar.gz
```

or, if `RESULTS_DIR` is changed, the archive is created from that submission
directory by `8_check_and_prepare_data_for_upload.py`.

## Reproducing The Full Pipeline

1. Prepare `./mvtec_ad_2`.
2. Place the DINOv3 repository under `./dinov3`.
3. Place the DINOv3 ViT-L/16 weight under `./pre_weights`.
4. Review `run_vand.sh` and set:

```bash
DATA_PATH="./mvtec_ad_2_aug"
RESULTS_DIR="./results"
TOTAL_EPOCHS=10
```

5. Run:

```bash
bash run_vand.sh
```

Runtime is approximately 6 hours on one RTX 3090. Temporary disk usage can be
around 110 GB, excluding the original dataset archive.

## Class-Agnostic Design Notes

The intended pipeline uses:

```text
- one shared architecture for all categories
- one shared pretrained backbone choice for all categories
- one shared preprocessing procedure
- one shared inference procedure
- one shared postprocessing procedure
```

The category name is used to select the corresponding dataset folder and output
folder. It should not be used to select a different architecture, backbone, or
manual category-specific processing logic.

No manual category-specific prompts are used.

Synthetic anomaly generation is not used in this pipeline.

Thresholding must be documented carefully. For VAND regular submission, do not
use `test_public`, `test_private`, or `test_private_mixed` data for threshold
optimization unless the official rules explicitly allow it.

## Acknowledgements

This project builds on ideas and code from:

```text
INP-Former: https://github.com/luow23/INP-Former
ISVL: https://github.com/ISVL119/isvl
DINOv3: https://github.com/facebookresearch/dinov3
```

Please cite and follow the licenses of the original projects and pretrained
weights.

## License

This code is intended for non-commercial research use for the VAND Industrial
Track.

Our challenge-specific modifications and pipeline code are released under
CC BY-NC 4.0 where permitted.

The pretrained DINOv3 backbone, INP-Former code, ISVL code, VAND/MVTec
submission utilities, and MVTec AD 2 dataset are governed by their respective
licenses. Please review those licenses before redistribution or reuse.
