# self supervised multimodal multi-task learning network
import math
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['CLGSI']

class CLGSI(nn.Module):
    def __init__(self, args):
        super(CLGSI, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.relu = nn.ReLU()

        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        audio_len, video_len = args.seq_lens[1:]
        self.audio_model = AuVi_Encoder(audio_in,args.a_encoder_heads,args.a_encoder_layers,audio_len,args.device)
        self.video_model = AuVi_Encoder(video_in,args.v_encoder_heads,args.v_encoder_layers,video_len,args.device)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.post_text_dim + args.post_audio_dim + args.post_video_dim, args.post_fusion_dim)
        self.GLFK = GLFK(enlager=args.fusion_filter_nums)
        skip_connection_length = args.post_text_dim + args.post_audio_dim + args.post_video_dim + args.post_text_dim
        self.skip_connection_BatchNorm = nn.BatchNorm1d(skip_connection_length)
        self.skip_connection_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_skip_connection = nn.Linear(skip_connection_length,args.post_fusion_dim*3)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim*3, args.post_fusion_dim*2)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim*2, 1)
        self.fusionBatchNorm = nn.BatchNorm1d(args.post_fusion_dim*2)

        # the aligned layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.text_skip_net = unimodal_skip_net(input_channel = args.text_out, enlager = args.post_text_dim, reduction = args.skip_net_reduction)   #跟对齐后的向量长度保持一致
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        # self.textBatchNorm = nn.BatchNorm1d(args.post_text_dim)


        # the aligned layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.audio_skip_net = unimodal_skip_net(input_channel = args.audio_out, enlager = args.post_text_dim, reduction = args.skip_net_reduction)  # 跟对齐后的向量长度保持一致
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        # self.audioBatchNorm = nn.BatchNorm1d(args.post_audio_dim)

        # the aligned layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.video_skip_net = unimodal_skip_net(input_channel = args.video_out, enlager = args.post_text_dim, reduction = args.skip_net_reduction)  # 跟对齐后的向量长度保持一致
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        # self.vedioBatchNorm = nn.BatchNorm1d(args.post_video_dim)


    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video


        text = self.text_model(text)
        audio = self.audio_model(audio)
        video = self.video_model(video)

        # # text
        text_h = self.post_text_dropout(text[:,0,:])  #取出BERT的[cls]
        text_skip = self.text_skip_net(text)     #取出BERT的最后一层隐藏层所有向量
        text_h = self.relu(self.post_text_layer_1(text_h))




        # audio
        audio_h = self.post_audio_dropout(audio[:,-1,:])  #取出transformer的最后一个结果
        audio_skip = self.audio_skip_net(audio)        #取出transformer的最后一层隐藏层所有向量
        audio_h = self.relu(self.post_audio_layer_1(audio_h))



        # vision
        video_h = self.post_video_dropout(video[:,-1,:])  #取出transformer的最后一个结果
        video_skip = self.video_skip_net(video)          #取出transformer的最后一层隐藏层所有向量
        video_h = self.relu(self.post_video_layer_1(video_h))



        # fusion
        # fusion_h = torch.cat([text_h, audio_h, video_h], dim=-1)
        fusion_h = torch.cat([text_h.unsqueeze(-1), audio_h.unsqueeze(-1), video_h.unsqueeze(-1)], dim=-1)
        fusion_h = self.GLFK(fusion_h)


        #skip-connection
        fusion_h = torch.cat([fusion_h, text_skip, audio_skip, video_skip], dim=-1)
        fusion_h = self.skip_connection_dropout(fusion_h)
        fusion_h = self.post_fusion_layer_skip_connection(fusion_h)


        # classifier-fusion
        x_f = self.relu(self.post_fusion_layer_2(fusion_h))
        x_f = self.fusionBatchNorm(x_f)
        output_fusion = self.post_fusion_layer_3(x_f)


        res = {
            'M': output_fusion,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res


class AuVi_Encoder(nn.Module):
    def __init__(self, hidden_size, nhead=1, num_layers=1, max_length = 1, device=None):
        '''
        Args:
            hidden_size: hidden layer dimension
            num_layers: the number of layers of transformer.
            nhead: the number of mult-head attention
        Output:
            (return value in forward) a tensor of shape (batch_size,sequence_len,hidden_size)
        '''
        super(AuVi_Encoder, self).__init__()
        self.position_embbeding = PositionalEncoding(hidden_size, 0.1,  device, max_length)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)


    def forward(self, x):
        '''
        x: (batch_size, sequence_len, hidden_size)
        '''

        x = self.position_embbeding(x)
        output = self.transformer_encoder(x)

        return output

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[ :, :-1]

        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # return self.dropout(x)
        return x



class GLFK(nn.Module):
    def __init__(self, enlager = 48):

        super(GLFK, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1),
            # nn.ReLU(),
            nn.Conv2d(1, enlager//2, kernel_size=(1, 1), stride=1),
            nn.PReLU(),
            nn.Conv2d(enlager//2, enlager, kernel_size=(1, 1), stride=1),
            nn.Conv2d(enlager, 1, kernel_size=(1, 1), stride=1),
            nn.Tanh()
        )


    def forward(self,x):
        output = x.unsqueeze(1)
        output = self.fc(output)
        output = output.squeeze()
        return output

class unimodal_skip_net(nn.Module):
    def __init__(self, input_channel, enlager, reduction=8):    #best maybe is 8

        super(unimodal_skip_net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel//reduction, enlager, bias=False),
            nn.Sigmoid()
        )



    def forward(self,x):
        # output = x.unsqueeze(2)
        output = x.transpose(2,1)
        output = output.unsqueeze(2)
        output = self.avg_pool(output).squeeze()
        output = self.fc(output)
        output = output.squeeze()
        return output
    
    
