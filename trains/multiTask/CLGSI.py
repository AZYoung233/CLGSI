import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from models.contrastive_loss import contrastive_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt


logger = logging.getLogger('MSA')

class CLGSI():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }



        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [               #设置需要更新的参数
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        total_steps = len(dataloader['train'])*self.args.warm_up_epochs   #大致的一个训练step数

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        saved_labels = {}
        # init labels
        # logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        losses = []
        valid_losses = []
        valid_F1 = []
        lr = []
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0     #评价阈值的初始化
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            CL_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()      #在训练1个batch之后停止梯度清0，当新的epoch来临时才清0
                    left_epochs -= 1                #这么做相当于把batch_size扩大为（N-1）*batch_size，其中N为一个epoch中的batch数

                    # optimizer.zero_grad()
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:   #分别读取出MVTA的label存入字典
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # compute loss
                    gamma = self.args.gamma  #loss weight
                    loss = 0.0
                    contrastive_loss_function = contrastive_loss(self.args.datasetName, self.args.device,self.args.dividing_line)
                    contrastive_loss1 = 0.0

                    loss = self.l1_loss(outputs['M'], self.label_map[self.name_map['M']][indexes], \
                                               indexes=indexes, mode=self.name_map['M'])
                    contrastive_loss1 = contrastive_loss_function(outputs,  self.label_map[self.name_map['M']][indexes] )

                    loss = loss + gamma*contrastive_loss1

                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    CL_loss += contrastive_loss1.item()

                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        scheduler.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            CL_loss = CL_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f  CL_loss:%.4f  " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss,CL_loss))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            losses.append(train_loss)
            # for m in self.args.tasks:
            #     pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            #     train_results = self.metrics(pred, true)
            #     logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            valid_losses.append(val_results['Loss'])
            valid_F1.append(cur_valid)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # early stop
            if epochs - best_epoch >= self.args.early_stop:     #如果比best_epoch再过了early_stop轮之后还没有出现新的best_epoch，就停止训练
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                # self.loss_plt(losses,valid_losses,valid_F1)
                # self.lr_plt(lr)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.l1_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results
    
    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss

    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
    
    def loss_plt(self,loss,valid_losses,valid_F1):
        train_x = range(len(loss))
        train_y = loss
        valid_x = range(len(valid_losses))
        valid_y = valid_losses
        F1_x = range(len(valid_F1))
        F1_y = valid_F1
        save_path = os.path.join(self.args.res_save_dir, \
                                 f'{self.args.datasetName}-{self.args.train_mode}.jpg')
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(train_x , train_y, label='Train')
        axs[0].set_ylabel('Train-Loss')
        # axs[0].set_ylim([0, max(train_y) * 1.2])  # 设置train loss的y轴范围
        # axs[0].set_yticks(np.arange(0, max(train_y) + 0.1, (max(train_y) - min(train_y)) / 5))  # 设置train loss的y轴刻度密度
        axs[1].plot(valid_x, valid_y, label='Valid')
        axs[1].set_ylabel('Valid-Loss')
        axs[1].legend(loc='upper right')
        axs[2].plot(F1_x, F1_y, label='valid')
        axs[2].set_ylabel('Valid-F1 ')
        # axs[1].set_ylim([min(valid_y) - 0.05, max(valid_y) + 0.05])  # 设置valid F1的y轴范围
        # axs[1].set_yticks(
        #     np.arange(min(valid_y), max(valid_y) + 0.01, (max(valid_y) - min(valid_y)) / 5))  # 设置valid F1的y轴刻度密度
        axs[2].legend(loc='upper right')
        plt.xlabel('epoch')
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(save_path,dpi=300, bbox_inches='tight',transparent=True)
        # 保存图形并关闭窗口
        plt.close()
        # 保存本地
        plt.show()

    def lr_plt(self,lr):
        plt.plot(np.arange(len(lr)), lr)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Warm-up Learning Rate Schedule')
        plt.show()
