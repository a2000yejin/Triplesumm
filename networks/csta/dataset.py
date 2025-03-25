import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import math
import h5py
import json
from torch.nn.utils.rnn import pad_sequence

from networks.a2summ.helpers.bbox_helper import get_loc_label, get_ctr_label
from networks.a2summ.helpers.vsumm_helper import get_keyshot_summ

def match_frame_num(feature, target_feature):
    frame_num_diff = feature.shape[0] - target_feature.shape[0]
    if frame_num_diff < 0:
        zero_padding = np.zeros((-frame_num_diff, feature.shape[1]))
        feature = np.vstack([zero_padding, feature])
        #print(f"{video} padding {-1*frame_num_diff}", flush = True)
                
    elif frame_num_diff > 0:
        feature = feature[frame_num_diff:, :]
        #print(f"{video} truncate {frame_num_diff}", flush = True)
    
    assert feature.shape[0] == target_feature.shape[0]

    return feature

# Model: PGL_SUM, CSTA
class MrSumDataset(Dataset):
    def __init__(self, feature_path, data_type, vis_data, mode, combine_with_vis):
        # self.mode = mode
        self.feature_dataset = feature_path
        self.visual_dataset = '/home/yejin/data/dataset/MR.HiSum/feature_h5/video.h5'
        self.gt_dataset = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/mrsum_with_features_gtsummary_modified.h5'
        self.split_file = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/custom_split.json'
        self.data_type = data_type
        self.vis_data = vis_data
        self.mode = mode
        self.metadata_file = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/metadata.csv'
        
        self.feature_data = h5py.File(self.feature_dataset, 'r')
        self.gt_data = h5py.File(self.gt_dataset, 'r')
        self.visual_data = h5py.File(self.visual_dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.splits = json.loads(f.read())

        self.metadata = pd.read_csv(self.metadata_file)

        self.combine_with_vis = combine_with_vis

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        # self.len = len(self.data[self.mode+'_keys']['Sports'])
        self.len = len(self.splits[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        # video_name = self.data[self.mode + '_keys']['Sports'][index]
        youtube_id = self.splits[self.mode + '_keys'][index]
        video_id = self.metadata[self.metadata['youtube_id'] == youtube_id]['video_id'].values[0]

        d = {}
        d['video_name'] = video_id
        d['gtscore'] = torch.Tensor(np.array(self.gt_data[video_id]['gtscore']))

        if self.vis_data == 'clip':
            #d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
            feature = self.feature_data[youtube_id][self.data_type][()]  # [num_frames, emb_dim]

            feature = match_frame_num(feature, d['gtscore'])
            feature = torch.Tensor(feature)
            
        elif self.vis_data == 'googleNet':
            feature = torch.Tensor(np.array(self.gt_data[video_id]['features']))
            assert feature.shape[0] == d['gtscore'].shape[0], "feature frame num is different!"

        if self.combine_with_vis:
            vis_feature = self.visual_data[youtube_id]['video'][()]
            vis_feature = match_frame_num(vis_feature, d['gtscore'])
            vis_feature = torch.Tensor(vis_feature)
            feature = torch.cat((feature, vis_feature), dim=1)

        d['features'] = feature

        #if self.mode != 'train':
        n_frames = d['features'].shape[0]
        cps = np.array(self.gt_data[video_id]['change_points'])
        d['n_frames'] = np.array(n_frames)
        d['picks'] = np.array([i for i in range(n_frames)])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
        if d['change_points'][-1][1] != n_frames:
            d['n_frame_per_seg'][-1] += 1 # Add 1 for the last change point frame
        d['gt_summary'] = np.expand_dims(np.array(self.gt_data[video_id]['gt_summary']), axis=0)
        
        return d

class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore= [],[],[]
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator', flush=True)

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len = max(list(map(lambda x: x.shape[0], features)))
        # max_len = 300

        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        
        frame_feat = pad_sequence(features, batch_first=True) #torch.Size([batch_size, max_len, input_dim])
        gtscore = pad_sequence(gtscore, batch_first=True) #torch.Size([batch_size, max_len])


        # Pad sequence to length 300
        frame_feat = self.pad_to_max_length(frame_feat, max_len)
        gtscore = self.pad_to_max_length(gtscore, max_len)


        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
                        'gt_summary': gt_summary}
        return batch_data

    def pad_to_max_length(self, padded_sequence, max_length):
        # Pad
        if padded_sequence.size(1) < max_length:
            # 필요한 경우 추가 패딩
            padding_size = max_length - padded_sequence.size(1)
            padding = torch.zeros((padded_sequence.size(0), padding_size, *padded_sequence.size()[2:]), dtype=padded_sequence.dtype)
            padded_sequence = torch.cat([padded_sequence, padding], dim=1)
        return padded_sequence



