#!/bin/bash

#SBATCH --job-name=testing    # Submit a job named "example"
#SBATCH --partition=a4000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1        # Use 0 GPU
#SBATCH --time=14-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=5G              # cpu memory size
#SBATCH --cpus-per-task=4        # cpu 개수 (gpu당 최대 8개)
#SBATCH --output=/home/yejin/data/projects/yejin/VideoSum/Triplesumm/slurm_log/testing.out  # 스크립트 실행 결과 std output을 저장할 파일 이름
#SBATCH --nodelist=node05        # 사용할 노드 이름

cd /home/yejin/data/projects/yejin/VideoSum/Triplesumm

ml purge
ml load cuda/12.1            # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate Triplesumm         # Activate your conda environment

StartTime=$(date +%s)

python testing.py \
    --model A2Summ \
    --mix_type 'vt' \
    --vis_feature_path /home/yejin/data/dataset/MR.HiSum/feature_h5/video.h5 \
    --text_feature_path /home/yejin/data/dataset/MR.HiSum/feature_h5/transcripts.h5 \
    --audio_feature_path /home/yejin/data/dataset/MR.HiSum/feature_h5/audio.h5 \
    --timestamp_data_path /home/yejin/data/dataset/MR.HiSum/feature_h5/timestamp.h5 \
    --gt_path /home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/mrsum_with_features_gtsummary_modified.h5 \
    --split_file_path /home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/custom_split.json \
    --tensorboard_path /home/yejin/data/projects/yejin/VideoSum/Triplesumm/tensorboard \
    --train true \
    --batch_size 32 \
    --epochs 1 \
    --lr 5e-5 \
    --early_stop_by None \
    --save_results True \
    --tag $SLURM_JOB_NAME \
    --weight_decay 1e-7
    # --l2_reg 1e-7 \
    # --gamma 0.99 \
    

EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."




