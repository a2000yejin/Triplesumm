import os
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import pandas as pd
import json
import math
from tqdm import tqdm

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
    
    assert feature.shape[0] == target_feature.shape[0], "feature frame num is different!"

    return feature

def time_to_frame(time_str, fps = 1):
    h, m, s = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    frame_index = total_seconds * fps
    return frame_index


class VideoSummDataset(object):
    def __init__(self, vis_data, additional_data_path, additional_data_type, timestamp_data_path, mode):
        #visual data
        self.vis_data = vis_data #clip or googleNet
        self.clip_dataset = '/home/yejin/data/dataset/MR.HiSum/feature_h5/video.h5'
        self.gt_dataset = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/mrsum_with_features_gtsummary_modified.h5'
        
        self.split_file = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/custom_split.json'
        
        self.clip_data = h5py.File(self.clip_dataset, 'r')
        self.gt_data = h5py.File(self.gt_dataset, 'r')

        #additional modality data
        self.additional_data_path = additional_data_path #audio or transcripts
        self.additional_data_type = additional_data_type
        self.timestamp_data_path = timestamp_data_path #'/home/yejin/data/dataset/MR.HiSum/feature_h5/timestamp.h5'

        self.additional_data = h5py.File(self.additional_data_path, 'r')
        self.timestamp = h5py.File(self.timestamp_data_path, 'r')
        
        #split
        self.mode = mode #train or val or test
        self.metadata_file = '/home/yejin/data/projects/yejin/VideoSum/Triplesumm/data/metadata.csv'
        
        with open(self.split_file, 'r') as f:
            self.splits = json.loads(f.read())

        self.metadata = pd.read_csv(self.metadata_file)

    def __len__(self):
        self.len = len(self.splits[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        youtube_id = self.splits[self.mode + '_keys'][index]
        video_id = self.metadata[self.metadata['youtube_id'] == youtube_id]['video_id'].values[0]

        d = {}
        d['video_name'] = video_id
        d['gtscore'] = torch.Tensor(np.array(self.gt_data[video_id]['gtscore'])) # [num_frames]
        
        # visual data
        if self.vis_data == 'clip':
            video = self.clip_data[youtube_id]['video'][()]  # [num_frames, emb_dim]
        elif self.vis_data == 'googleNet':
            video = torch.Tensor(np.array(self.gt_data[video_id]['features']))
        video = match_frame_num(video, d['gtscore'])
        video = torch.Tensor(video)
        d['video'] = video
        n_frames = d['video'].shape[0] #length after padding
        d['n_frames'] = np.array(n_frames)

        # text or audio data
        text = self.additional_data[youtube_id][self.additional_data_type][()] # [num_frames, emb_dim]
        text = match_frame_num(text, d['gtscore'])
        text = torch.Tensor(text)
        d['text'] = text
        n_sentences = text.shape[0] #length after padding
        d['n_sentences'] = n_sentences

        # gt data about visual
        cps = np.array(self.gt_data[video_id]['change_points'])
        d['picks'] = np.array([i for i in range(n_frames)])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
        if d['change_points'][-1][1] != n_frames:
            d['n_frame_per_seg'][-1] += 1 # Add 1 for the last change point frame
        d['gt_summary'] = np.expand_dims(np.array(self.gt_data[video_id]['gt_summary']), axis=0)

        # gtscore -= gtscore.min()
        # gtscore /= gtscore.max()

        keyshot_summ, gtscore_upsampled = get_keyshot_summ(d['gtscore'], d['change_points'], d['n_frames'], d['n_frame_per_seg'], d['picks'])
        target = keyshot_summ
        # target = keyshot_summ[::15]

        video_cls_label = target
        video_loc_label = get_loc_label(target)
        video_ctr_label = get_ctr_label(target, video_loc_label)

        d['video_cls_label'] = torch.from_numpy(video_cls_label)
        d['video_loc_label'] = torch.from_numpy(video_loc_label)
        d['video_ctr_label'] = torch.from_numpy(video_ctr_label)

        
        frame_sentence_ratio = int(math.ceil(n_frames / n_sentences))
        text_cls_label = np.zeros((n_sentences), dtype=bool)
        for j in range(n_sentences):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, n_frames)
            if video_cls_label[start_frame: end_frame].any():
                text_cls_label[j] = True

        text_loc_label = get_loc_label(text_cls_label)
        text_ctr_label = get_ctr_label(text_cls_label, text_loc_label)

        d['text_cls_label'] = torch.from_numpy(text_cls_label)
        d['text_loc_label'] = torch.from_numpy(text_loc_label)
        d['text_ctr_label'] = torch.from_numpy(text_ctr_label)
        
        time_index = self.timestamp[youtube_id][()]
        time_index = time_index.decode('utf-8')
        time_index = json.loads(time_index)
        video_to_text_mask = torch.zeros((n_frames, n_sentences), dtype=torch.long)
        text_to_video_mask = torch.zeros((n_sentences, n_frames), dtype=torch.long)
        for j in range(len(time_index)):
            start_time, end_time = time_index[j]['start_time'], time_index[j]['end_time']
            start_frame, end_frame = time_to_frame(start_time), time_to_frame(end_time)
            video_to_text_mask[start_frame:end_frame, start_frame:end_frame] = 1
            text_to_video_mask[start_frame:end_frame, start_frame:end_frame] = 1
        d['video_to_text_mask'] = video_to_text_mask
        d['text_to_video_mask'] = text_to_video_mask

        mask_video = torch.ones(n_frames, dtype=torch.long)
        mask_text = torch.ones(n_sentences, dtype=torch.long)
        d['mask_video'] = mask_video
        d['mask_text'] = mask_text
        
        return d


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class BatchCollator(object):
    def __call__(self, batch):
        video_name, gtscore, video, n_frames, text, n_sentences = [], [], [], [], [], []
        picks, change_points, n_frame_per_seg, gt_summary = [], [], [], []
        video_cls_label, video_loc_label, video_ctr_label, text_cls_label, text_loc_label, text_ctr_label = [], [], [], [], [], []
        video_to_text_mask, text_to_video_mask = [], []
        mask_video, mask_text = [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                gtscore.append(data['gtscore'])
                video.append(data['video'])
                n_frames.append(data['n_frames'])
                text.append(data['text'])
                n_sentences.append(data['n_sentences'])
                picks.append(data['picks'])
                change_points.append(data['change_points'])
                n_frame_per_seg.append(data['n_frame_per_seg'])
                gt_summary.append(data['gt_summary'])
                video_cls_label.append(data['video_cls_label'])
                video_loc_label.append(data['video_loc_label'])
                video_ctr_label.append(data['video_ctr_label'])
                text_cls_label.append(data['text_cls_label'])
                text_loc_label.append(data['text_loc_label'])
                text_ctr_label.append(data['text_ctr_label'])
                video_to_text_mask.append(data['video_to_text_mask'])
                text_to_video_mask.append(data['text_to_video_mask'])
                mask_video.append(data['mask_video'])
                mask_text.append(data['mask_text'])

        except:
            print('Error in batch collator', flush=True)

        #padding
        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], video)))
        max_len = max(list(map(lambda x: x.shape[0], video)))

        video = pad_sequence(video, batch_first = True)
        text = pad_sequence(text, batch_first = True)
        gtscore = pad_sequence(gtscore, batch_first = True)

        mask_video = pad_sequence(mask_video, batch_first = True)
        mask_text = pad_sequence(mask_text, batch_first = True)

        video_label = pad_sequence(video_cls_label, batch_first = True)
        text_label = pad_sequence(text_cls_label, batch_first = True)

        batch_data = {'video_name': video_name, 'video': video, 'text': text,
                      'mask_video': mask_video, 'mask_text': mask_text,
                      'video_label': video_label, 'text_label': text_label,
                      'video_to_text_mask': video_to_text_mask, 'text_to_video_mask': text_to_video_mask}

        return batch_data


    
#def my_collate_fn(batch):
#    batched_output_list = []
#    for i in range(len(batch[0])):
#        batched_output = [item[i] for item in batch]
#        batched_output_list.append(batched_output)
#    return batched_output_list
