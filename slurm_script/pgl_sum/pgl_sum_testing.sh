#!/bin/bash

#SBATCH --job-name=pgl_sum_mrhisum_visual_testing    # Submit a job named "example"
#SBATCH --partition=a4000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1        # Use 0 GPU
#SBATCH --time=14-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=5G              # cpu memory size
#SBATCH --cpus-per-task=4        # cpu 개수 (gpu당 최대 8개)
#SBATCH --output=/home/yejin/data/projects/yejin/VideoSum/Triplesumm/slurm_log/pgl_sum/pgl_sum_mrhisum_visual_testing.out  # 스크립트 실행 결과 std output을 저장할 파일 이름
#SBATCH --nodelist=node05        # 사용할 노드 이름

cd /home/yejin/data/projects/yejin/VideoSum/Triplesumm

ml purge
ml load cuda/11.3            # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate Triplesumm         # Activate your conda environment

StartTime=$(date +%s)

python main.py \
    --train True \
    --dataset mrhisum \
    --feature_path /home/yejin/data/dataset/MR.HiSum/feature_h5/video.h5 \
    --data_type video \
    --model PGL_SUM \
    --batch_size 5 \
    --epochs 1 \
    --tag $SLURM_JOB_NAME \
    --l2_reg 1e-7 \
    --lr 5e-5 \
    --gamma 0.99 \
    --train_val False \
    --individual False \
    --save_results True \
    --early_stop_by None

EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."




