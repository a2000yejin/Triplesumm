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

def time_to_frame(time_str, fps = 1):
    h, m, s = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    frame_index = total_seconds * fps
    return frame_index

class TripleSummDataset(Dataset):
    def __init__(self, vis_feature_path, text_feature_path, audio_feature_path, timestamp_data_path, gt_path, split_file_path, mode):
        self.vis_feature_path = vis_feature_path
        self.text_feature_path = text_feature_path
        self.audio_feature_path = audio_feature_path
        self.timestamp_data_path = timestamp_data_path
        self.gt_path = gt_path  #mrsum_with_features_gtsummary_modified.h5
        self.split_file_path = split_file_path  #train/val/test split json file
        self.mode = mode  #train/val/test

        #video_id (ex. video_1)와 youtube_id 맵핑한 csv
        self.metadata_file_path = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/metadata.csv'


        #get features
        self.vis_feature = h5py.File(self.vis_feature_path, 'r')
        self.text_feature = h5py.File(self.text_feature_path, 'r')
        self.audio_feature = h5py.File(self.audio_feature_path, 'r')
        self.gt = h5py.File(self.gt_path, 'r')

        #get timestamp
        self.timestamp = h5py.File(self.timestamp_data_path, 'r')

        #get split
        with open(self.split_file_path, 'r') as f:
            self.splits = json.loads(f.read())

        #get metadata
        self.metadata = pd.read_csv(self.metadata_file_path)

    def __len__(self):
        self.len = len(self.splits[self.mode + '_keys'])
        return self.len

    def __getitem__(self, index):
        youtube_id = self.splits[self.mode + '_keys'][index] # for feature
        video_id = self.metadata[self.metadata['youtube_id'] == youtube_id]['video_id'].values[0]  # for gt

        d = {}

        #==================basic data=====================
        #metadata
        d['video_id'] = video_id

        #gtscore
        d['gtscore'] = torch.Tensor(np.array(self.gt[video_id]['gtscore']))

        #features
        vis_feature = self.vis_feature[youtube_id]['video'][()]
        vis_feature = torch.Tensor(match_frame_num(vis_feature, d['gtscore']))
        d['vis_feature'] = vis_feature
        d['n_frames'] = vis_feature.shape[0]

        text_feature = self.text_feature[youtube_id]['transcripts'][()]
        text_feature = torch.Tensor(match_frame_num(text_feature, d['gtscore']))
        d['text_feature'] = text_feature

        audio_feature = self.audio_feature[youtube_id]['audio'][()]
        audio_feature = torch.Tensor(match_frame_num(audio_feature, d['gtscore']))
        d['audio_feature'] = audio_feature
        
        #gt data about visual
        cps = np.array(self.gt[video_id]['change_points'])
        d['change_points'] = cps

        d['picks'] = np.array([i for i in range(d['n_frames'])])

        d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
        if d['change_points'][-1][1] != d['n_frames']:
            d['n_frame_per_seg'][-1] += 1 # Add 1 for the last change point frame

        d['gt_summary'] = np.expand_dims(np.array(self.gt[video_id]['gt_summary']), axis=0)


        #==================data for A2Summ=====================
        keyshot_summ, gtscore_upsampled = get_keyshot_summ(d['gtscore'], d['change_points'], d['n_frames'], d['n_frame_per_seg'], d['picks'])
        target = keyshot_summ
        # target = keyshot_summ[::15]

        #visual cls, loc, ctr
        vis_cls_label = target
        vis_loc_label = get_loc_label(target)
        vis_ctr_label = get_ctr_label(target, vis_loc_label)

        d['vis_cls_label'] = torch.from_numpy(vis_cls_label)
        d['vis_loc_label'] = torch.from_numpy(vis_loc_label)
        d['vis_ctr_label'] = torch.from_numpy(vis_ctr_label)

        #text cls, loc, ctr과 video_to_text_mask, text_to_video_mask
        time_index = self.timestamp[youtube_id][()]
        time_index = time_index.decode('utf-8')
        time_index = json.loads(time_index)
        text_cls_label = np.zeros((d['n_frames']), dtype=bool)
        vis_to_text_mask = torch.zeros((d['n_frames'], d['n_frames']), dtype=torch.long) #text is expanded to match n_frames
        text_to_vis_mask = torch.zeros((d['n_frames'], d['n_frames']), dtype=torch.long)
        d['n_sents'] = len(time_index) #number of unique sentences
        text_segment_mask = np.zeros((d['n_sents'], d['n_frames']))
        for j in range(len(time_index)):
            start_time, end_time = time_index[j]['start_time'], time_index[j]['end_time']
            start_frame, end_frame = time_to_frame(start_time), time_to_frame(end_time)
            text_segment_mask[j, start_frame:end_frame] = 1
            
            if d['vis_cls_label'][start_frame:end_frame].any():
                text_cls_label[start_frame:end_frame] = True

            vis_to_text_mask[start_frame:end_frame, start_frame:end_frame] = 1
            text_to_vis_mask[start_frame:end_frame, start_frame:end_frame] = 1

        d['text_segment_mask'] = text_segment_mask

        text_loc_label = get_loc_label(text_cls_label)
        text_ctr_label = get_ctr_label(text_cls_label, text_loc_label)
        d['text_cls_label'] = torch.from_numpy(text_cls_label)
        d['text_loc_label'] = torch.from_numpy(text_loc_label)
        d['text_ctr_label'] = torch.from_numpy(text_ctr_label)

        d['vis_to_text_mask'] = vis_to_text_mask
        d['text_to_vis_mask'] = text_to_vis_mask

        mask = torch.ones(d['n_frames'], dtype=torch.long)

        d['mask'] = mask

        return d

class BatchCollator(object):
    def __call__(self, batch):
        video_id, gtscore, vis_feature, n_frames, text_feature, n_sents, audio_feature = [], [], [], [], [], [], []
        picks, change_points, n_frame_per_seg, gt_summary = [], [], [], []
        vis_cls_label, vis_loc_label, vis_ctr_label, text_cls_label, text_loc_label, text_ctr_label = [], [], [], [], [], []
        text_segment_mask = [],
        vis_to_text_mask, text_to_vis_mask = [], []
        mask = []

        try:
            for data in batch:
                video_id.append(data['video_id'])
                gtscore.append(data['gtscore'])
                vis_feature.append(data['vis_feature'])
                n_frames.append(data['n_frames'])
                text_feature.append(data['text_feature'])
                n_sents.append(data['n_sents'])
                audio_feature.append(data['audio_feature'])
                picks.append(data['picks'])
                change_points.append(data['change_points'])
                n_frame_per_seg.append(data['n_frame_per_seg'])
                gt_summary.append(data['gt_summary'])
                vis_cls_label.append(data['vis_cls_label'])
                vis_loc_label.append(data['vis_loc_label'])
                vis_ctr_label.append(data['vis_ctr_label'])
                text_cls_label.append(data['text_cls_label'])
                text_loc_label.append(data['text_loc_label'])
                text_ctr_label.append(data['text_ctr_label'])
                print(type(text_segment_mask), type(data['text_segment_mask']))
                text_segment_mask.append(data['text_segment_mask'])
                vis_to_text_mask.append(data['vis_to_text_mask'])
                text_to_vis_mask.append(data['text_to_vis_mask'])
                mask.append(data['mask'])

        except:
            print('Error in batch collator', flush=True)

        #padding
        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], vis_feature)))
        max_len = max(list(map(lambda x: x.shape[0], vis_feature)))

        vis_feature = pad_sequence(vis_feature, batch_first = True)
        text_feature = pad_sequence(text_feature, batch_first = True)
        audio_feature = pad_sequence(audio_feature, batch_first = True)
        gtscore = pad_sequence(gtscore, batch_first = True)

        mask = pad_sequence(mask, batch_first = True)

        vis_label = pad_sequence(vis_cls_label, batch_first = True)
        text_label = pad_sequence(text_cls_label, batch_first = True)

        batch_data = {'video_id': video_id, 'vis_feature': vis_feature, 'text_feature': text_feature, 'audio_feature': audio_feature,
                      'gtscore' : gtscore,
                      'mask': mask, 
                      'n_frames': n_frames, 'picks' : picks, 'n_frame_per_seg' : n_frame_per_seg, 'change_points' : change_points,
                      'n_sents': n_sents, 'text_segment_mask':text_segment_mask,
                      'gt_summary' : gt_summary, 'vis_label': vis_label, 'text_label': text_label,
                      'vis_to_text_mask': vis_to_text_mask, 'text_to_vis_mask': text_to_vis_mask}

        return batch_data