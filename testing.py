import os
import sys
import torch
import argparse
import json
import h5py
from tqdm import tqdm

from configs import get_config
from datasets import TripleSummDataset, BatchCollator
from torch.utils.data import DataLoader
from networks.csta import cstaSolver
from networks.a2summ import A2SummSolver

import numpy as np
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #모든 모델의 공통된 파라메터 먼저 받기
    parser.add_argument('--model', type=str, required=True, help="Summarization model name")
    parser.add_argument('--mix_type', type=str, required=True, choices = ['v', 't', 'a', 'vt', 'ta', 'va', 'vta'], help='modalities to use')
    parser.add_argument('--vis_feature_path', type=str, required=True, help="Visual feature h5 file path")
    parser.add_argument('--text_feature_path', type=str, required=True, help="Text feature h5 file path")
    parser.add_argument('--audio_feature_path', type=str, required=True, help="Audio feature h5 file path")
    parser.add_argument('--timestamp_data_path', type=str, required=True, help="Timestamp h5 file path")
    parser.add_argument('--gt_path', type=str, required=True, help="Gt h5 file path")
    parser.add_argument('--split_file_path', type=str, required=True, help="Train/val/test split json file path")
    parser.add_argument('--train', type=str, required=True, help="Whether to train")
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--early_stop_by', type=str, required=True, help="f1score/map50/map15")
    parser.add_argument('--save_results', type=bool, required=True)
    parser.add_argument('--tag', type=str, required=True)
    
    # parser.add_argument('--l2_reg', type=float, required=True)
    # parser.add_argument('--gamma', type=float, required=True, help="Gamma for CSTA exponential learning rate scheduler")
    
    args, unknown = parser.parse_known_args()
    
    unknown_dict = {}
    for i in range(0, len(unknown), 2):
        if unknown[i].startswith("--"):  # "--"로 시작하는 경우만 처리
            key = unknown[i].lstrip("--")  # "--" 제거
            value = unknown[i + 1] if i + 1 < len(unknown) else None
            unknown_dict[key] = value

    config = get_config(args.model)
    config.__dict__.update(vars(args))
    config.__dict__.update(unknown_dict)

    if config.model in ['CSTA', 'PGL_SUM']:
        input_dim = 768 * len(args.mix_type)
        config.input_dim = input_dim

    print(config.__dict__)

    train_data = TripleSummDataset(
        vis_feature_path = config.vis_feature_path,
        text_feature_path = config.text_feature_path,
        audio_feature_path = config.audio_feature_path,
        timestamp_data_path = config.timestamp_data_path,
        gt_path = config.gt_path,
        split_file_path = config.split_file_path,
        mode = 'train'
    )
    val_data = TripleSummDataset(
        vis_feature_path = config.vis_feature_path,
        text_feature_path = config.text_feature_path,
        audio_feature_path = config.audio_feature_path,
        timestamp_data_path = config.timestamp_data_path,
        gt_path = config.gt_path,
        split_file_path = config.split_file_path,
        mode = 'val'
    )
    test_data = TripleSummDataset(
        vis_feature_path = config.vis_feature_path,
        text_feature_path = config.text_feature_path,
        audio_feature_path = config.audio_feature_path,
        timestamp_data_path = config.timestamp_data_path,
        gt_path = config.gt_path,
        split_file_path = config.split_file_path,
        mode = 'test'
    )

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=BatchCollator())
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=BatchCollator())
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=BatchCollator())
    

    #=====================데이터로더 확인=============================
    # num_batches = int(len(train_loader))
    # iterator = iter(train_loader)
    # for batch_idx in tqdm(range(num_batches)):
    #     data = next(iterator)
    #     for key in data.keys():
    #         if key in ['video_id', 'n_frames', 'picks', 'n_frame_per_seg', 'change_points', 'gt_summary']:
    #             print(f"{key} : {len(data[key])}")
    #         elif key in ['vis_to_text_mask', 'text_to_vis_mask']:
    #             print(f"{key} : length = {len(data[key])}, shape of first = {data[key][0].shape}")
    #         else:
    #             print(f"{key} : {data[key].detach().shape}")
    #     break


    #=====================csta, pgl-sum 확인=============================
    # csta_solver = cstaSolver(config, train_loader, val_loader, test_loader, ckpt_path=config.ckpt_path)
    # csta_solver.build()
    # f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path, srho_save_ckpt_path, ktau_save_ckpt_path = csta_solver.train()

    #=====================a2summ 확인=============================
    a2summ_solver = A2SummSolver(config, train_loader, val_loader, test_loader, ckpt_path=config.ckpt_path)
    a2summ_solver.build()
    f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path, srho_save_ckpt_path, ktau_save_ckpt_path = a2summ_solver.train()


    #create timestamp h5py file
    # timestamp_folder = '/home/yejin/data/dataset/MR.HiSum/transcripts_postprocessed'
    # output_h5_file = '/home/yejin/data/dataset/MR.HiSum/feature_h5/timestamp.h5'

    # with h5py.File(output_h5_file, 'w') as h5f:
    #     for filename in tqdm(os.listdir(timestamp_folder), desc="Processing JSON files"):
    #         if filename.endswith(".json"):  # JSON 파일만 처리
    #             video_id = filename[:-5]  # 확장자 .json 제거하여 video_id 추출
    #             file_path = os.path.join(timestamp_folder, filename)
                
    #             # JSON 파일 읽기
    #             with open(file_path, "r", encoding="utf-8") as f:
    #                 json_data = json.load(f)
                
    #             # JSON 데이터를 문자열로 변환하여 HDF5에 저장
    #             json_str = json.dumps(json_data)
    #             h5f.create_dataset(video_id, data=json_str)

    # with h5py.File(output_h5_file, 'r') as h5f:
    #     video_id = '__b4x3D3bxw'
    #     if video_id in h5f:
    #         json_data = json.loads(h5f[video_id][()].decode())
    #         print(json_data)