import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple


from .positional_encoding import FixedPositionalEncoding,LearnablePositionalEncoding,RelativePositionalEncoding,ConditionalPositionalEncoding

def check_mask(mask):
    if torch.all((mask == 0) | (mask == 1)):
        print("mask는 0과 1로만 이루어져 있습니다.")
    else:
        print("mask에 0과 1 이외의 값이 포함되어 있습니다.")

# Edit GoogleNet by replacing last parts with adaptive average pooling layers
class GoogleNet_Att(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        init_weights: Optional[bool] = None
    ) -> None:
        super().__init__()
        conv_block = BasicConv2d
        inception_block = Inception

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaskedMaxPool(3, 2, True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaskedMaxPool(3, 2, True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaskedMaxPool(3, 2, True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaskedMaxPool(2, 2, True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x: Tensor, mask: Tensor, n_frame) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        
        mask = mask.unsqueeze(1).unsqueeze(-1)
        x, mask = self.conv1(x, mask)
        x, mask = self.maxpool1(x, mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.conv3(x, mask)
        x, mask = self.maxpool2(x, mask)

        x, mask = self.inception3a(x, mask)
        x, mask = self.inception3b(x, mask)
        x, mask = self.maxpool3(x, mask)
        x, mask = self.inception4a(x, mask)

        x, mask = self.inception4b(x, mask)
        x, mask = self.inception4c(x, mask)
        x, mask = self.inception4d(x, mask)

        x, mask = self.inception4e(x, mask)
        x, mask = self.maxpool4(x, mask)
        x, mask = self.inception5a(x, mask)
        x, mask = self.inception5b(x, mask)

        ##############################################################################
        # The place I edit to resize feature maps, and to handle various lengths of input videos
        ##############################################################################
        #print(f"x before avgpool: {x.shape}", flush=True) #torch.Size([batch_size, emb_dim, height, width])
        #print(f"mask before avgpool: {mask.shape}", flush=True) #torch.Size([batch_size, 1, height, 1])

        self.avgpool = MaskedAdaptiveAvgPool((n_frame+1,1))
        x, mask = self.avgpool(x, mask)  
        #print(f"x after avgpool: {x.shape}", flush=True) #torch.Size([batch_size, emb_dim, n_frame+1, 1])
        #print(f"mask after avgpool: {mask.shape}", flush=True) #torch.Size([batch_size, 1, n_frame+1, 1])
        
        x = x.squeeze(dim=3) 
        #print(f"after squeeze: {x.shape}", flush=True) #torch.Size([batch_size, emb_dim, n_frame+1])
        x = x.permute(0, 2, 1) 
        #print(f"after permute: {x.shape}", flush=True) #torch.Size([batch_size, n_frame+1, emb_dim])
        return x, mask
    
    def forward(self, x: Tensor, mask: Tensor, n_frame):
        ##############################################################################
        # Takes the number of frames to handle various lengths of input videos
        ##############################################################################
        # x: torch.Size([batch_size, channel=3, frame_num+1, self.dim=1024])
        # mask: torch.Size([batch_size, frame_num+1])
        
        x, mask = self._forward(x, mask, n_frame)
        return x, mask

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        maxpool = MaskedMaxPool

        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2_0 = conv_block(in_channels, ch3x3red, kernel_size=1)
        self.branch2_1 = conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)

        self.branch3_0 = conv_block(in_channels, ch5x5red, kernel_size=1)
        self.branch3_1 = conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)

        self.branch4_0 = maxpool(3, 1, ceil_mode=True, padding=1)
        self.branch4_1 = conv_block(in_channels, pool_proj, kernel_size=1)

    def _forward(self, x: Tensor, mask: Tensor) -> List[Tensor]:
        branch1, mask = self.branch1(x, mask)
        
        branch2, mask = self.branch2_0(x, mask)
        branch2, mask = self.branch2_1(branch2, mask)

        branch3, mask = self.branch3_0(x, mask)
        branch3, mask = self.branch3_1(branch3, mask)

        branch4, mask = self.branch4_0(x, mask)
        branch4, mask = self.branch4_1(branch4, mask)

        outputs = [branch1, branch2, branch3, branch4]

        return outputs, mask
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        outputs, mask = self._forward(x, mask)
        return torch.cat(outputs, 1), mask

class MaskedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x, mask):
        #print(f"input x in MaskedBatchNorm2d: {x.shape}", flush=True) #torch.Size([batch_size, channel, width, height])
        #print(f"input mask in MaskedBatchNorm2d: {mask.shape}", flush=True) # torch.Size([batch_size, 1, width, 1])

        #compute mean and variance only for non-padded regions
        x_sum = x.sum(dim=(0, 2, 3), keepdim=True)
        masked_count = mask.sum(dim=(0, 2, 3), keepdim=True).clamp(min=1e-5)
        mean = x_sum / masked_count    
        
        variance = ((x - mean) ** 2 * mask).sum(dim=(0, 2, 3), keepdim=True) / masked_count

        #print(f"mean: {mean.shape}", flush=True) #torch.Size([1, channel, 1, 1])
        #print(f"variance: {variance.shape}", flush=True) #torch.Size([1, channel, 1, 1])

        if self.training:
            self.running_mean = ((1 - self.momentum) * self.running_mean.view(1, -1, 1, 1) + self.momentum * mean).squeeze()
            self.running_var = ((1 - self.momentum) * self.running_var.view(1, -1, 1, 1) + self.momentum * variance).squeeze()
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            variance = self.running_var
            #print(f"mean: {mean.shape}", flush=True) #torch.Size([channel])
            #print(f"variance: {variance.shape}", flush=True) #torch.Size([channel])

        #normalize
        x_norm = (x - mean.view(1, -1, 1, 1)) / (variance.view(1, -1, 1, 1) + self.eps).sqrt() # torch.Size([batch_size, num_features, width, height])

        weight = self.weight.view(1, -1, 1, 1)  # [1, num_features, 1, 1]
        bias = self.bias.view(1, -1, 1, 1)    # [1, num_features, 1, 1]

        out = x_norm * weight + bias

        return out * mask, mask

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = MaskedBatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.conv(x)

        mask = F.conv2d(mask,
                        weight=torch.ones(1, 1, *self.conv.kernel_size).to(x.device),
                        stride=self.conv.stride,
                        padding=self.conv.padding,
                        bias=None)
        mask = (mask > 0).float()

        x, mask = self.bn(x, mask)

        x = F.relu(x, inplace=True)
        
        x = x * mask
        return x, mask

class MaskedMaxPool(nn.Module):
    def __init__(self, kernel_size=3, stride=2, ceil_mode=True, padding = 0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, ceil_mode=ceil_mode, padding = padding)
        self.maskpool = nn.MaxPool1d(kernel_size, stride, ceil_mode=ceil_mode, padding=padding)

    def forward(self, x: Tensor, mask: Tensor):
        x = x * mask + (1 - mask) * (-1e9)
        #print(x, flush=True)
        x = self.pool(x)

        mask = self.maskpool(mask.squeeze(1).squeeze(-1))
        mask = mask.unsqueeze(1).unsqueeze(-1)
        mask = (mask > 0).float()

        x = x* mask
        return x, mask

class MaskedAdaptiveAvgPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        self.maskAvgPool = nn.AdaptiveAvgPool1d(output_size[0])

    def forward(self, x:Tensor, mask:Tensor):
        #print(f"x in final merge: {x.shape}", flush=True)  #torch.Size([batch_size, 1, num_frame+1, emb_dim])
        #print(f"mask in final merge: {mask.shape}", flush=True) #torch.Size([batch_size, 1, num_frame+1, 1])

        pooled_x = self.avgpool(x * mask)
        pooled_mask = self.maskAvgPool(mask.squeeze(-1)).unsqueeze(-1)

        #print(f"pooled_x: {pooled_x.shape}", flush=True)
        #print(f"pooled_mask: {pooled_mask.shape}", flush=True)

        pooled_x = pooled_x / (pooled_mask + 1e-8)

        return pooled_x, pooled_mask

class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        
        x_masked = x.masked_fill(mask == 0, -1e9)
        softmax_x = F.softmax(x_masked, dim=self.dim)

        return softmax_x

##############################################################################
#   Define our proposed model
##############################################################################
class CSTA_GoogleNet(nn.Module):
    def __init__(self,
                 model_name,
                 Scale,
                 Softmax_axis,
                 Balance,
                 Positional_encoding,
                 Positional_encoding_shape,
                 Positional_encoding_way,
                 Dropout_on,
                 Dropout_ratio,
                 Classifier_on,
                 CLS_on,
                 CLS_mix,
                 key_value_emb,
                 Skip_connection,
                 Layernorm,
                 input_dim,
                 batch_size,
                 dim=1024):
        super().__init__()
        self.googlenet = GoogleNet_Att()

        self.model_name = model_name
        self.Scale = Scale
        self.Softmax_axis = Softmax_axis
        self.Balance = Balance

        self.Positional_encoding = Positional_encoding
        self.Positional_encoding_shape = Positional_encoding_shape
        self.Positional_encoding_way = Positional_encoding_way
        self.Dropout_on = Dropout_on
        self.Dropout_ratio = Dropout_ratio

        self.Classifier_on = Classifier_on
        self.CLS_on = CLS_on
        self.CLS_mix = CLS_mix

        self.key_value_emb = key_value_emb
        self.Skip_connection = Skip_connection
        self.Layernorm = Layernorm
 
        self.input_dim = input_dim
        self.dim = dim
        self.batch_size = batch_size

        if self.Positional_encoding is not None:
            if self.Positional_encoding=='FPE':
                self.Positional_encoding_op = FixedPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='RPE':
                self.Positional_encoding_op = RelativePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='LPE':
                self.Positional_encoding_op = LearnablePositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    dim=self.dim
                )
            elif self.Positional_encoding=='CPE':
                self.Positional_encoding_op = ConditionalPositionalEncoding(
                    Positional_encoding_shape=self.Positional_encoding_shape,
                    Positional_encoding_way=self.Positional_encoding_way,
                    dim=self.dim
                )
            elif self.Positional_encoding is None:
                pass
            else:
                raise

        if self.Positional_encoding_way=='Transformer':
            self.Positional_encoding_embedding = nn.Linear(in_features=self.dim, out_features=self.dim)
        elif self.Positional_encoding_way=='PGL_SUM' or self.Positional_encoding_way is None:
            pass 
        else:
            raise
        
        if self.Dropout_on:
            self.dropout = nn.Dropout(p=float(self.Dropout_ratio))
        
        if self.Classifier_on:
            self.linear1 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.LayerNorm(normalized_shape=self.dim, eps=1e-6))

            self.linear2 = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=1),
                nn.Sigmoid())

            for name,param in self.named_parameters():
                if name in ['linear1.0.weight','linear2.0.weight']:
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif name in ['linear1.0.bias','linear2.0.bias']:
                    nn.init.constant_(param, 0.1)
        else:
            self.gap = nn.AdaptiveAvgPool1d(1)
        
        if self.CLS_on:
            self.CLS = nn.Parameter(torch.zeros(self.batch_size,3,1,1024))

        if self.key_value_emb is not None:
            if self.key_value_emb.lower()=='k':
                self.key_embedding = nn.Linear(in_features=1024,out_features=self.dim)
            elif self.key_value_emb.lower()=='v':
                self.value_embedding = nn.Linear(in_features=self.dim,out_features=self.dim)
            elif ''.join(sorted(self.key_value_emb.lower()))=='kv':
                self.key_embedding = nn.Linear(in_features=1024,out_features=self.dim)
                if self.model_name=='GoogleNet_Attention':
                    self.value_embedding = nn.Linear(in_features=1024,out_features=self.dim)
            else:
                raise

        if self.Layernorm:
            if self.Skip_connection=='KC':
                self.layernorm1 = MaskedBatchNorm2d(num_features=1)
            elif self.Skip_connection=='CF':
                self.layernorm2 = MaskedBatchNorm2d(num_features=1)
            elif self.Skip_connection=='IF':
                self.layernorm3 = MaskedBatchNorm2d(num_features=1)
            elif self.Skip_connection is None:
                pass
            else:
                raise
        
        self.match_emb = nn.Linear(self.input_dim, 1024)

    def forward(self, x, mask): 
        #print(f"input x: {x.shape}", flush=True) #torch.Size([batch_size, channel=3, frame_num=max_len, emb_dim])
        #print(f"input mask: {mask.shape}", flush=True) #torch.Size([batch_size, frame_num=max_len])
        #orig_mask = mask.clone()

        # Take the number of frames
        batch_size = x.shape[0]
        n_frame = x.shape[2]
        #print(f"n_frame: {n_frame}", flush=True)
        

        #print(f"frame_num: {x.size(2)}", flush=True)
        if self.input_dim != self.dim:
            x = self.match_emb(x.reshape(-1,self.input_dim))
            x = x.view(-1, 3, n_frame, 1024)
            x = x * mask.unsqueeze(1).unsqueeze(-1)
            #print(f"x after match_emb: {x.shape}", flush=True) # torch.Size([batch_size, channel=3, frame_num, 1024])
            #print(f"mask after match_emb: {mask.shape}", flush=True) #torch.Size([batch_size, frame_num])

        
        
        
        # Linear projection if using CLS token as transformer ways
        if self.Positional_encoding_way=='Transformer':
            x = self.Positional_encoding_embedding(x)
        # Stack CLS token
        if self.CLS_on:
            if batch_size != self.batch_size:
                x = torch.cat((self.CLS[:batch_size, :, :, :],x),dim=2)
            else:
                x = torch.cat((self.CLS, x), dim=2)

            #Add valid token in mask for CLS token
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask =  torch.cat([cls_mask, mask], dim=1)

            #print(f"x after cls token: {x.shape}", flush=True) # [batch_size, channel=3, frame_num+1, 1024]
            #print(f"mask after cls token: {mask.shape}", flush=True) # [batch_size, frame_num+1]
            CT_adjust = MaskedAdaptiveAvgPool((n_frame, self.dim))
        
        #print(f"feature after cls token: {x.shape}", flush=True) #torch.Size([batch_size, channel=3, frame_num+1, 1024])
        
        # Positional encoding (Transformer ways)
        if self.Positional_encoding_way=='Transformer':
            if self.Positional_encoding is not None:
                x = self.Positional_encoding_op(x)
            # Dropout (Transformer ways)
            if self.Dropout_on:
                x = self.dropout(x)
        elif self.Positional_encoding_way=='PGL_SUM' or self.Positional_encoding_way is None:
            pass
        else:
            raise

        # Key Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['k','kv']:
            key = self.key_embedding(x)
            key = key * mask.unsqueeze(1).unsqueeze(-1)
        elif self.key_value_emb is None:
            key = x
        else:
            raise
        
        #print(f"key after key_value_emb: {key.shape}", flush=True) #torch.Size([batch_size, channel=3, frame_num+1, self.dim=1024])
        #print(f"mask after key_value_emb: {mask.shape}", flush=True) #torch.size([batch_size, frame_num+1])
        
        # CNN as attention algorithm
        x_att, mask_att = self.googlenet(key, mask, n_frame)
        #print(f"x_att: {x_att.shape}", flush=True) #torch.Size([batch_size, frame_num+1, emb_dim])
        #print(f"mask_att: {mask_att.shape}", flush=True) #torch.Size([batch_size, 1, frame_num+1, 1])
        
        # Skip connection (KC)
        if self.Skip_connection is not None:
            if self.Skip_connection=='KC':
                x_att = x_att + key[:, 0, :, :]
                #print(f"before layernorm: {x_att.shape}", flush=True) # torch.Size([batch_size, frame_num+1, emb_dim])
                if self.Layernorm:
                    x_att, mask_att = self.layernorm1(x_att.unsqueeze(1), mask_att)
                    x_att = x_att.squeeze(1)
            elif self.Skip_connection in ['CF','IF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        #print(f"x_att after skip connection: {x_att.shape}", flush=True) # [batch_size, frame_num+1, emb_dim]
        #print(f"mask_att after skip connection: {mask_att.shape}", flush=True) # [batch_size, 1, frame_num+1, 1]

        # Combine CLS token (CNN)
        if self.CLS_on:
            if self.CLS_mix=='CNN':
                x_att = CT_adjust(x_att.unsqueeze(1)).squeeze(1)  # [batch_size, frame_num, dim=1024]
                x = CT_adjust(x.unsqueeze(1)).squeeze(1) # [batch_size, frame_num, dim=1024]
            elif self.CLS_mix in ['SM','Final']:
                pass
            else:
                raise
        else:
            pass

        
        # Scaling factor
        if self.Scale is not None:
            if self.Scale=='D':
                scaling_factor = x_att.shape[2]
            elif self.Scale=='T':
                scaling_factor = x_att.shape[1]
            elif self.Scale=='T_D':
                scaling_factor = x_att.shape[1] * x_att.shape[2]
            else:
                raise
            scaling_factor = scaling_factor ** 0.5
            x_att = x_att / scaling_factor
        elif self.Scale is None:
            pass
        
        # Positional encoding (PGL-SUM ways)
        if self.Positional_encoding_way=='PGL_SUM':
            if self.Positional_encoding is not None:
                x_att = self.Positional_encoding_op(x_att, mask_att)
        elif self.Positional_encoding_way=='Transformer' or self.Positional_encoding_way is None:
            pass
        else:
            raise
        #print(f"x_att after positional encoding: {x_att.shape}", flush=True) ## [batch_size, frame_num+1, dim=1024]
        # softmax_axis
        if self.Softmax_axis=='T':
            temporal_attention = MaskedSoftmax(dim=1)(x_att, mask_att)
        elif self.Softmax_axis=='D':
            spatial_attention = F.softmax(x_att,dim=2)
        elif self.Softmax_axis=='TD':
            temporal_attention = MaskedSoftmax(dim=1)(x_att, mask_att.squeeze(1))
            spatial_attention = F.softmax(x_att,dim=2)
        elif self.Softmax_axis is None:
            pass
        else:
            raise

        

        # Combine CLS token for softmax outputs (SM)
        if self.CLS_on:
            if self.CLS_mix=='SM':
                if self.Softmax_axis=='T':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(1)).squeeze(1)
                elif self.Softmax_axis=='D':
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(1)).squeeze(1)
                elif self.Softmax_axis=='TD':
                    temporal_attention = CT_adjust(temporal_attention.unsqueeze(1)).squeeze(1)
                    spatial_attention = CT_adjust(spatial_attention.unsqueeze(1)).squeeze(1)
                elif self.Softmax_axis is None:
                    pass
                else:
                    raise
            elif self.CLS_mix in ['CNN','Final']:
                pass
            else:
                raise
        else:
            pass
        
        # Dropout (PGL-SUM ways)
        if self.Dropout_on and self.Positional_encoding_way=='PGL_SUM':
            if self.Softmax_axis=='T':
                temporal_attention = self.dropout(temporal_attention)
            elif self.Softmax_axis=='D':
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis=='TD':
                temporal_attention = self.dropout(temporal_attention)
                spatial_attention = self.dropout(spatial_attention)
            elif self.Softmax_axis is None:
                pass
            else:
                raise
        
        #print(f"temporal_attention after dropout: {temporal_attention.shape}", flush=True)
        #print(f"spatial_attention after dropout: {spatial_attention.shape}", flush=True)

        # Value Embedding
        if self.key_value_emb is not None and self.key_value_emb.lower() in ['v','kv']:
            if self.model_name=='GoogleNet_Attention':
                x_out = self.value_embedding(x[:, 0, :, :])
                #print(f"value: {x_out.shape}", flush=True) # [batch_size, frame_num+1, dim]
                #print(f"mask to be multiplied with value: {mask.shape}", flush=True) # [batch_size, frame_num+1]
  
                x_out = x_out * mask.unsqueeze(-1)
                #print(f"value after multiplying with mask: {x_out.shape}", flush=True) # [batch_size, frame_num+1, dim]
                
            elif self.model_name=='GoogleNet':
                x_out = x_att
            else:
                raise
        elif self.key_value_emb is None:
            if self.model_name=='GoogleNet':
                x_out = x_att
            elif self.model_name=='GoogleNet_Attention':
                x_out = x
            else:
                raise
        else:
            raise
        
        # Combine CLS token for CNN outputs (SM)
        if self.CLS_on:
            if self.CLS_mix=='SM':
                x_out = CT_adjust(x_out.unsqueeze(1)).squeeze(1) # [batch_size, frame_num, dim]

        # Apply Attention maps to input frame features
        if self.Softmax_axis=='T':
            x_out = x_out * temporal_attention
        elif self.Softmax_axis=='D':
            x_out = x_out * spatial_attention
        elif self.Softmax_axis=='TD':
            #print(f"apply attention: {x_out.shape}, {temporal_attention.shape}, {spatial_attention.shape}", flush=True)
            batch_size, T,D = x_out.shape
            adjust_frame = T/D
            adjust_dimension = D/T
            if self.Balance=='T':
                x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
            elif self.Balance=='D':
                x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
            elif self.Balance=='BD':
                if T>D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                elif T<D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                elif T==D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance=='BU':
                if T>D:
                    x_out = x_out * temporal_attention * adjust_frame + x_out * spatial_attention
                elif T<D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention * adjust_dimension
                elif T==D:
                    x_out = x_out * temporal_attention + x_out * spatial_attention
            elif self.Balance is None:
                x_out = x_out * temporal_attention + x_out * spatial_attention
            else:
                raise
        elif self.Softmax_axis is None:
            x_out = x_out * x_att
        else:
            raise
        
        # Skip Connection (CF)
        if self.Skip_connection is not None:
            if self.Skip_connection=='CF':
                if x_out.shape != x_att.shape:
                    x_att = CT_adjust(x_att.unsqueeze(1)).squeeze(1)
                x_out = x_out + x_att
                if self.Layernorm:
                    x_out = self.layernorm2(x_out.unsqueeze(1)).squeeze(1)
            elif self.Skip_connection in ['KC','IF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        
        # Skip Connection (IF)
        if self.Skip_connection is not None:
            if self.Skip_connection=='IF':
                x_out = x_out + x
                if self.Layernorm:
                    x_out = self.layernorm3(x_out.unsqueeze(1)).squeeze(1)
            elif self.Skip_connection in ['KC','CF']:
                pass
            else:
                raise
        elif self.Skip_connection is None:
            pass
        else:
            raise
        
        # Combine CLS token (Final)
        #print(f"value before final cls token combine: {x_out.shape}", flush=True) #torch.Size([batch_size, n_frames+1, emb_dim])

        if self.CLS_on:
            if self.CLS_mix=='Final':
                x_out, mask_out = CT_adjust(x_out.unsqueeze(1), mask.unsqueeze(1).unsqueeze(-1))
                x_out = x_out.squeeze(1)

                #print(f"x_out final merge: {x_out.shape}", flush=True) # torch.Size([batch_size, n_frames, emb_dim])
                #print(f"mask_out final merge: {mask_out.shape}", flush=True) # torch.Size([batch_size, 1, n_frames, 1])

            elif self.CLS_mix in ['CNN','SM']:
                pass
            else:
                raise
        else:
            pass
        
        # Classifier
        if self.Classifier_on:
            x_out = self.linear1(x_out)
            #print(f"x_out after linear1: {x_out.shape}", flush=True) #torch.Size([batch_size, n_frames, emb_dim])

            x_out = self.linear2(x_out)
            #print(f"x_out after linear2: {x_out.shape}", flush=True) #torch.Size([batch_size, n_frames, 1])
            #print(f"mask_out: {mask_out.shape}", flush=True) #torch.Size([batch_size, 1, n_frames, 1])

            x_out = x_out * mask_out.squeeze(1)
            x_out = x_out.squeeze()
            
            #print(f"is orig_mask and mask_out the same: {orig_mask == mask_out.squeeze()}")
            #print(f"classifier: {x_out.shape}", flush=True) # torch.Size([batch_size, n_frames])
        else:
            x_out = self.gap(x_out)
            x_out = x_out.squeeze()

        return x_out