# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.pgl_sum.attention import SelfAttention


class MultiAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=None, heads=1, fusion=None):
        """ Class wrapping the MultiAttention part of PGL-SUM; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(MultiAttention, self).__init__()

        # Global Attention, considering differences among all frames
        self.attention = SelfAttention(input_size=input_size, output_size=output_size,
                                       freq=freq, pos_enc=pos_enc, heads=heads)

        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                # Local Attention, considering differences among the same segment with reduce hidden size
                self.local_attention.append(SelfAttention(input_size=input_size, output_size=output_size//num_segments,
                                                          freq=freq, pos_enc=pos_enc, heads=4))
        self.permitted_fusions = ["add", "mult", "avg", "max"]
        self.fusion = fusion
        if self.fusion is not None:
            self.fusion = self.fusion.lower()
            assert self.fusion in self.permitted_fusions, f"Fusion method must be: {*self.permitted_fusions,}"

    def forward(self, x, mask=None):
        """ Compute the weighted frame features, based on the global and locals (multi-head) attention mechanisms.

        :param torch.Tensor x: Tensor with shape [T, input_size] containing the frame features.
        :return: A tuple of:
            weighted_value: Tensor with shape [T, input_size] containing the weighted frame features.
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        #print(f"x in MultiAttention: {x.shape}", flush=True) #[T, frame_num, input_size=emb_dim]

        weighted_value, attn_weights = self.attention(x, mask)  # global attention
        #print(f"global weighted value: {weighted_value.shape}", flush=True) # [batch_size, frame_num, emb_dim]
        #print(f"global attn weights: {attn_weights.shape}", flush=True) # [batch_size, frame_num, frame_num]


        if self.num_segments is not None and self.fusion is not None:
            B = x.shape[0]
            segment_size = math.ceil(x.shape[1] / self.num_segments)
            for segment in range(self.num_segments): # n_seg = 4
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[:, left_pos:right_pos]
                local_mask = None
                if mask is not None: 
                    local_mask = mask[:, left_pos:right_pos]
                weighted_local_value, attn_local_weights = self.local_attention[segment](local_x, local_mask)  # local attentions

                
                #print(f"weighted local value: {weighted_local_value.shape}", flush=True) # [batch_size, segment_size, emb_dim]
                #print(f"attn local weights: {attn_local_weights.shape}", flush=True) #[batch_size, segment_size, segment_size]

                # Normalize the features vectors
                weighted_value[:, left_pos:right_pos] = F.normalize(weighted_value[:, left_pos:right_pos].clone(), p=2, dim=2)
                weighted_local_value = F.normalize(weighted_local_value, p=2, dim=2)

                if self.fusion == "add":
                    weighted_value[:, left_pos:right_pos] += weighted_local_value
                elif self.fusion == "mult":
                    weighted_value[:, left_pos:right_pos] *= weighted_local_value
                elif self.fusion == "avg":
                    weighted_value[:, left_pos:right_pos] += weighted_local_value
                    weighted_value[:, left_pos:right_pos] /= 2
                elif self.fusion == "max":
                    weighted_value[:, left_pos:right_pos] = torch.max(weighted_value[left_pos:right_pos].clone(),
                                                                   weighted_local_value)

        return weighted_value, attn_weights


class PGL_SUM(nn.Module):
    def __init__(self, input_dim=768, feature_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=None, heads=1, fusion=None):
        """ Class wrapping the PGL-SUM model; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(PGL_SUM, self).__init__()

        self.input_dim = input_dim
        self.feature_size = feature_size
        self.match_emb = nn.Linear(self.input_dim, self.feature_size)

        self.attention = MultiAttention(input_size=self.feature_size, output_size=output_size, freq=freq,
                                        pos_enc=pos_enc, num_segments=num_segments, heads=heads, fusion=fusion)
        self.linear_1 = nn.Linear(in_features=self.feature_size, out_features=self.feature_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=self.feature_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, vis_feature, text_feature, audio_feature, mask, mix_type):
        """ 
        :param torch.Tensor frame_features: Tensor of shape [BS, N, input_dim] containing the frame features
        """
        #print(f"frame features: {frame_features.shape}", flush=True) #[batch_size, frame_num, emb_dim]
        #print(f"mask: {mask.shape}", flush=True) #[batch_size, frame_num]
        if mix_type == 'v':
            x = vis_feature
        elif mix_type == 't':
            x = text_feature
        elif mix_type == 'a':
            x = audio_feature
        elif mix_type == 'vt':
            x = torch.cat((vis_feature, text_feature), dim = -1)
        elif mix_type == 'va':
            x = torch.cat((vis_feature, audio_feature), dim = -1)
        elif mix_type == 'ta':
            x = torch.cat((text_feature, audio_feature), dim = -1)
        elif mix_type == 'vta':
            x = torch.cat((vis_feature, text_feature, audio_feature), dim = -1)
        
        bs, n, dim = x.shape
        #print(f"Before projection: {x.shape}", flush=True) # [batch_size, n_frames, input_size]

        if self.input_dim != self.feature_size:
            x = self.match_emb(x.reshape(-1,self.input_dim))
            x = x.view(bs, n, self.feature_size)
            x = x * mask.unsqueeze(-1)
        #print(f"After projection: {x.shape}", flush=True) # [batch_size, n_frames, dim=1024]
        #print(f"Mask shape: {mask.shape}", flush=True) # [batch_size, n_frames]
        
        residual = x
        weighted_value, attn_weights = self.attention(x, mask=mask)
        y = weighted_value + residual
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(bs, -1)
        return y, attn_weights


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = PGL_SUM(input_size=256, output_size=256, num_segments=3, fusion="Add").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
