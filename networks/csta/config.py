import json 
import argparse
import numpy as np


def build_config():
    from configs import Config, str2bool
    parser = argparse.ArgumentParser("This script is used for PGL_SUM/CSTA summarization task.")
    
    # Training & evaluation
    parser.add_argument('--model', type = str, default = 'MLP', help = 'the name of the model')
    parser.add_argument('--epochs', type = int, default = 200, help = 'the number of training epochs')
    parser.add_argument('--lr', type = float, default = 5e-5, help = 'the learning rate')
    parser.add_argument('--l2_reg', type = float, default = 1e-4, help = 'l2 regularizer')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--batch_size', type = int, default = 256, help = 'the batch size')
    parser.add_argument('--tag', type = str, default = 'dev', help = 'A tag for experiments')
    parser.add_argument('--ckpt_path', type = str, default = None, help = 'checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
    parser.add_argument('--save_results', type=bool, default=False, help='when save results')
    parser.add_argument('--early_stop_by', type=str, default=None)
    parser.add_argument('--tensorboard_path', type = str, default = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/tensorboard')

    # Data Config
    parser.add_argument('--vis_feature_path', type=str, help="path to visual feature h5 file")
    parser.add_argument('--text_feature_path', type=str, help="path to text feature h5 file")
    parser.add_argument('--audio_feature_path', type=str, help="path to audio feature h5 file")
    parser.add_argument('--timestamp_data_path', type=str, help="path to timestamp feature h5 file")
    parser.add_argument('--gt_path', type=str, help="path to gt h5 file")
    parser.add_argument('--split_file_path', type=str, help="path to split file")
    parser.add_argument("--max_v_l", type=int, default=-1)
    parser.add_argument('--mix_type', type=str, default='v')
    parser.add_argument('--input_dim', type=int, default=768)
    

    # Model Config
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_scores', default=300, type=int, help="Number of query slots")
    parser.add_argument('--input_dropout', default=0.5, type=float, help="Dropout applied in input")
    parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
    parser.add_argument("--K", type=int, default=200, help="Quantization size for scores")
    parser.add_argument("--denoiser", type=str, default='DiT', choices=['DiT', 'Transformer_dec', 'latentmlp'], help="Denoiser model")
    parser.add_argument("--p_uncond", type=float, default=0.0, help="Probability of sampling from unconditional")
    parser.add_argument("--w", type=float, default=0.1, help="weight for unconditional sampling")
    parser.add_argument("--ema", type=str2bool, default=False, help="use EMA for denoiser")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="decay for EMA")
    parser.add_argument("--sigmoid_temp", type=float, default=1.0, help="temperature for sigmoid")
    parser.add_argument("--eps", type=float, default=1e-2, help="eps for logit")
    parser.add_argument("--scores_embed", type=str, default='learned', choices=['learned', 'sinusoidal'], help="score embedding")
    parser.add_argument("--clamp", type=str2bool, default=False, help="clamp scores")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--contrastive_align_loss", action="store_true", help="Disable contrastive_align_loss between matched query spans and the text.")
    parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
    parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'], 
                        help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
    parser.add_argument("--lw_saliency", type=float, default=4.,
                            help="weight for saliency loss, set to 0 will ignore")
    parser.add_argument("--saliency_margin", type=float, default=0.2)
        
        
    # * Matcher
    parser.add_argument('--set_cost_span', default=10, type=float, help="L1 span coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float, help="giou span coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=4, type=float, help="Class coefficient in the matching cost")

    # Loss coefficientsw
    parser.add_argument('--span_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--label_loss_coef', default=4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_align_loss_coef", default=0.02, type=float)
    parser.add_argument("--aux_loss_coef", default=0, type=float)
    parser.add_argument("--dec_loss_coef", default=1, type=float)

    args = parser.parse_args()

    kwargs = vars(args)
    config = Config(**kwargs)

    return config