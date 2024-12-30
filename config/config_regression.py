import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'clgsi': self.__CLGSI
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = '/home/sharing/disk3/dataset/multimodal-sentiment-dataset/StandardDatasets'
        root_dataset_dir = '/home/young/DL/multimodal_dataset/'
        tmp = {
            'mosi':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'mosei':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            },

            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232),  # (text, audio, video)
                    'feature_dims': (768, 25, 177),  # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            }
        }
        return tmp

    def __CLGSI(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,       # finetune BERT
                'save_labels': False,
                'dividing_line':0.4,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,       #5e-5 maybe is best
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 1e-3,
                    'learning_rate_other': 1e-2,
                    'weight_decay_bert': 0.01,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'fusion_filter_nums':16,     #24 is nice choice
                    'a_encoder_heads': 1,
                    'v_encoder_heads': 4,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'text_out': 768, 
                    'audio_out': 5,
                    'video_out': 20,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim': 64,
                    'post_audio_dim': 64,
                    'post_video_dim': 64,
                    'post_fusion_dropout': 0.2,
                    'post_text_dropout': 0.05,
                    'post_audio_dropout': 0.05,
                    'post_video_dropout': 0.05,
                    'skip_net_reduction': 4,
                    'warm_up_epochs': 110,
                    'gamma': 0.24,
                    'update_epochs': 1,
                    'early_stop': 8,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 128,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 25e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    # feature subNets
                    'fusion_filter_nums':16,     #16 is a nice choice
                    'a_encoder_heads': 2,
                    'v_encoder_heads': 5,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'text_out': 768, 
                    'audio_out': 74,
                    'video_out': 35,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 256,
                    'post_text_dim':128,
                    'post_audio_dim': 128,
                    'post_video_dim': 128,
                    'post_fusion_dropout': 0.1,      #0.1
                    'post_text_dropout': 0.01,
                    'post_audio_dropout': 0.01,
                    'post_video_dropout': 0.01,
                    'skip_net_reduction': 8,
                    'warm_up_epochs':120,
                    #loss weight   best：1
                    'gamma':0.33,                  #best is 46
                    'update_epochs': 1,
                    'early_stop': 8,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'fusion_filter_nums': 16,
                    'a_encoder_heads': 3,
                    'v_encoder_heads': 1,
                    'a_encoder_layers': 2,
                    'v_encoder_layers':2,
                    'text_out': 768, 
                    'audio_out': 33,
                    'video_out': 709,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 256,
                    'post_text_dim':128,
                    'post_audio_dim': 128,
                    'post_video_dim': 128,
                    'post_fusion_dropout': 0.1,  #0.1
                    'post_text_dropout': 0.4,
                    'post_audio_dropout': 0.4,
                    'post_video_dropout': 0.4,
                    'skip_net_reduction': 8,
                    'warm_up_epochs': 35,
                    'update_epochs': 1,
                    'early_stop': 8,
                    # loss weight
                    'gamma': 1,
                    # res
                    'H': 1.0
                },

            },
        }
        return tmp

    def get_config(self):
        return self.args