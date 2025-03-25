import json 
import argparse
import numpy as np


_DATASET_HYPER_PARAMS = {
    "mrhisum":{
        "seed": 12345,

        "input_dim_vis": 768, #input video embedding size
        "input_dim_text": 768, #input audio/text embedding size
        "num_hidden": 256,
        "num_layers": 2,

        "dropout_video": 0.1,
        "dropout_text": 0.1,
        "dropout_attn": 0.1,
        "dropout_fc": 0.5,

        "lambda_contrastive_inter": 0.0,
        "lambda_contrastive_intra": 0.0,
        "ratio": 4, #select (n_frames//ratio) nonkey frames for intra-sample contrastive loss
    }
} 

def build_config():
    from configs import Config, str2bool
    parser = argparse.ArgumentParser("This script is used for A2Summ summarization.")

    # Training & evaluation
    parser.add_argument('--model', type=str, default='A2Summ')
    parser.add_argument('--epochs', type = int, default = 200, help = 'the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tag', type = str, default = 'dev', help = 'A tag for experiments')
    parser.add_argument('--ckpt_path', type = str, default = None, help = 'checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
    parser.add_argument('--save_results', type=bool, default=False, help='when save results')
    parser.add_argument('--early_stop_by', type=str, default=None)
    parser.add_argument('--tensorboard_path', type = str, default = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/tensorboard')

    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', '-j', type=int, default=4)    
    
    # Data config
    parser.add_argument('--vis_feature_path', type=str, help="path to visual feature h5 file")
    parser.add_argument('--text_feature_path', type=str, help="path to text feature h5 file")
    parser.add_argument('--audio_feature_path', type=str, help="path to audio feature h5 file")
    parser.add_argument('--timestamp_data_path', type=str, help="path to timestamp feature h5 file")
    parser.add_argument('--gt_path', type=str, help="path to gt h5 file")
    parser.add_argument('--split_file_path', type=str, help="path to split file")
    parser.add_argument('--mix_type', type=str, default='v')

    # common model config
    parser.add_argument('--input_dim_vis', type=int, default=768)
    parser.add_argument('--input_dim_text', type=int, default=768)
    parser.add_argument('--num_hidden', type=int, default=128)
    
    # transformer config
    parser.add_argument('--dropout_video', type=float, default=0.1, help='pre_drop for video')
    parser.add_argument('--dropout_text', type=float, default=0.1, help='pre_drop for text')
    parser.add_argument('--dropout_attn', type=float, default=0.1, help='dropout for attention operation in transformer')
    parser.add_argument('--dropout_fc', type=float, default=0.5, help='dropout for final classification')
    parser.add_argument('--num_layers', type=int, default=1)

    # contrastive loss
    parser.add_argument('--lambda_contrastive_inter', type=float, default=0.0)
    parser.add_argument('--lambda_contrastive_intra', type=float, default=0.0)
    parser.add_argument('--ratio', type=int, default=16)


    args = parser.parse_args()
    args.seed = _DATASET_HYPER_PARAMS['mrhisum']["seed"]
    
    args.input_dim_vis = _DATASET_HYPER_PARAMS['mrhisum']["input_dim_vis"]
    args.input_dim_text = _DATASET_HYPER_PARAMS['mrhisum']["input_dim_text"]
    args.num_hidden = _DATASET_HYPER_PARAMS['mrhisum']["num_hidden"]
    args.num_layers = _DATASET_HYPER_PARAMS['mrhisum']["num_layers"]

    args.dropout_video = _DATASET_HYPER_PARAMS['mrhisum']["dropout_video"]
    args.dropout_text = _DATASET_HYPER_PARAMS['mrhisum']["dropout_text"]
    args.dropout_attn = _DATASET_HYPER_PARAMS['mrhisum']["dropout_attn"]
    args.dropout_fc = _DATASET_HYPER_PARAMS['mrhisum']["dropout_fc"]
    
    args.lambda_contrastive_inter = _DATASET_HYPER_PARAMS['mrhisum']["lambda_contrastive_inter"]
    args.lambda_contrastive_intra = _DATASET_HYPER_PARAMS['mrhisum']["lambda_contrastive_intra"]
    args.ratio = _DATASET_HYPER_PARAMS['mrhisum']["ratio"]

    kwargs = vars(args)
    config = Config(**kwargs)

    return config

