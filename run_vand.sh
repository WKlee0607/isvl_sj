#!/usr/bin/bash
#SBATCH -J vandcheck_dinov3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-g4
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm-%A.out

set -e  # 에러 발생 시 즉시 중단

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date +%Y%m%d_%H%M%S)"
echo "=========================================="

# -------------------------------------------------------
# 0. 로그 디렉토리 생성
# -------------------------------------------------------
mkdir -p logs
DATA_PATH="./mvtec_ad_2_aug"
TOTAL_EPOCHS=15
export DINOV3_REPO="${DINOV3_REPO:-./dinov3}"
export DINOV3_WEIGHTS="./pre_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# -------------------------------------------------------
# 1. 데이터 전처리
# -------------------------------------------------------
echo "[1/6] Data preprocessing..."
python 1_image_splitter.py

# -------------------------------------------------------
# 2. INP-Former 학습 (can, fabric, rice, sheet_metal, wallplugs, walnuts)
# -------------------------------------------------------
echo "[2/6] INP-Former training..."
python isvl.py --data_path "$DATA_PATH" --item_list can        --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list fabric     --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list rice       --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list sheet_metal --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list wallplugs  --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list walnuts    --total_epochs "$TOTAL_EPOCHS"

python isvl.py --data_path "$DATA_PATH" --item_list fruit_jelly --total_epochs "$TOTAL_EPOCHS"
python isvl.py --data_path "$DATA_PATH" --item_list vial        --total_epochs "$TOTAL_EPOCHS"


# -------------------------------------------------------
# 5. 추론 (val / test)
# -------------------------------------------------------
echo "[5/6] Inference..."

# INP-Former inference
python isvl.py --data_path "$DATA_PATH" --item_list can         --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list fabric      --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list rice        --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list sheet_metal --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list wallplugs   --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list walnuts     --total_epochs "$TOTAL_EPOCHS" --phase val

python isvl.py --data_path "$DATA_PATH" --item_list fruit_jelly --total_epochs "$TOTAL_EPOCHS" --phase val
python isvl.py --data_path "$DATA_PATH" --item_list vial        --total_epochs "$TOTAL_EPOCHS" --phase val



# -------------------------------------------------------
# 6. 후처리 및 제출 파일 준비
# -------------------------------------------------------
echo "[6/6] Postprocessing..."

python 2_image_reconstruction.py
python 3_replace_and_rename_folders.py
python 4_threshold_mapv2.py
python 5_post_image_process.py
python 7_convert_tiff_to_float16.py
python 8_check_and_prepare_data_for_upload.py "./results/"

echo "=========================================="
echo "Done: $(date +%Y%m%d_%H%M%S)"
echo "=========================================="
