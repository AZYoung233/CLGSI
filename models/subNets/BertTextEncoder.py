import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel,AutoTokenizer,AutoModel

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = AutoTokenizer
        model_class = AutoModel
        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        pretrained_model_en = 'bert-base-uncased'              #pretrained model select
        pretrained_model_cn = 'bert-base-chinese'
        if language == 'en':                 #虽然但是，这里并没有用到tokenizer，因为原始数据里面已经使用的bert
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_en, do_lower_case=True)
            self.model = model_class.from_pretrained(pretrained_model_en)
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_cn)
            self.model = model_class.from_pretrained(pretrained_model_cn)
        
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        # text = self.tokenizer(text,
        #                             padding='max_length',  # 如果样本长度不满足最大长度则填充
        #                             truncation=True,  # 截断至最大长度
        #                             max_length=self.max_len,
        #                             return_tensors='pt')  # 返回tensor格式

        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
    
if __name__ == "__main__":
    bert_normal = BertTextEncoder()
