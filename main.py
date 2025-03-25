import os
import sys
import torch
import argparse

from torch.utils.data import DataLoader
from networks.csta import cstaMrSumDataset, cstaBatchCollator, cstaSolver
from networks.a2summ import A2SummMrSumDataset, A2SummBatchCollator, A2SummSolver

from configs import get_config

import numpy as np
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--summModel', type=str, required=True)
    args = parser.parse_args()

    config = get_config(args.summModel)
    config.summModel = args.summModel
   
    if config.summModel in ['CSTA', 'PGL_SUM']:
        train_dataset = cstaMrSumDataset(feature_path=config.feature_path, data_type=config.data_type, vis_data=config.vis_data, mode='train', combine_with_vis=config.combine_with_vis)
        val_dataset = cstaMrSumDataset(feature_path=config.feature_path, data_type=config.data_type, vis_data=config.vis_data, mode='val', combine_with_vis=config.combine_with_vis)
        test_dataset = cstaMrSumDataset(feature_path=config.feature_path, data_type=config.data_type, vis_data=config.vis_data, mode='test', combine_with_vis=config.combine_with_vis)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=cstaBatchCollator())
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=cstaBatchCollator())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=cstaBatchCollator())
    
        solver = cstaSolver(config, train_loader, val_loader, test_loader, ckpt_path=config.ckpt_path)

    elif config.summModel == 'A2Summ':
        train_dataset = A2SummMrSumDataset(vis_data=config.vis_data, additional_data_path=config.additional_data_path, timestamp_data_path=config.timestamp_data_path, mode='train')
        val_dataset = A2SummMrSumDataset(vis_data=config.vis_data, additional_data_path=config.additional_data_path, timestamp_data_path=config.timestamp_data_path, mode='val')
        test_dataset = A2SummMrSumDataset(vis_data=config.vis_data, additional_data_path=config.additional_data_path, timestamp_data_path=config.timestamp_data_path, mode='test')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=A2SummBatchCollator())
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=A2SummBatchCollator())
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=A2SummBatchCollator())

        solver = A2SummSolver(config, train_loader, val_loader, test_loader, ckpt_path=config.checkpoint)

    solver.build()
    test_model_ckpt_path = None
    if config.train:
        try:
            best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path, best_srho_ckpt_path, best_ktau_ckpt_path = solver.train()
        except torch.cuda.OutOfMemoryError:
            print("🔥 OOM 에러 발생! GPU 메모리가 부족합니다.")
            torch.cuda.empty_cache()  # 캐시 비우기 (임시 해결책)
            sys.exit(1)
            
        solver.test(best_f1_ckpt_path)
        solver.test(best_map50_ckpt_path)
        solver.test(best_map15_ckpt_path)
        solver.test(best_srho_ckpt_path)
        solver.test(best_ktau_ckpt_path)
        
        if config.ema:
            print('EMA Testing')
            solver.test(best_f1_ckpt_path.replace('.pkl', '_ema.pkl'))
            solver.test(best_map50_ckpt_path.replace('.pkl', '_ema.pkl'))
            solver.test(best_map15_ckpt_path.replace('.pkl', '_ema.pkl'))
            solver.test(best_srho_ckpt_path.replace('.pkl', '_ema.pkl'))
            solver.test(best_ktau_ckpt_path.replace('.pkl', '_ema.pkl'))
    
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path == None:
            print("Trained model checkpoint required. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)
            if config.ema:
                print('EMA Testing')
                solver.test(test_model_ckpt_path.replace('.pkl', '_ema.pkl'))
    solver.writer.flush()
    solver.writer.close()
