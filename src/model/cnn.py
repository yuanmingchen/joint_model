#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(123456)

"""
CNN for Stance Classification
"""


class CNN(nn.Module):

    def __init__(self, embeddings, input_dim=100, hidden_dim=50, output_dim=2, max_len=40, dropout=0.5):
        super().__init__()
        self.embeddings = embeddings;
        self.max_len = max_len
        self.input_dim = input_dim
        self.emb = nn.Embedding(
            num_embeddings=embeddings.size(0),
            embedding_dim=embeddings.size(1),
            padding_idx=0
        )
        self.emb.weight = nn.Parameter(embeddings);

        '''
        加载预训练的词嵌入的方法
        方法1：
            self.emb.weight = nn.Parameter(embeddings);
        方法2：
            self.emb.weight.data.copy_=embeddings;
        '''
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, input_dim), stride=1,)
        self.hidden1 = nn.Linear(in_features=16, out_features=hidden_dim);
        self.out = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    # sen_batch:8*40
    def forward(self, sen_batch, sen_length=None):
        batch_size = len(sen_batch)
        sen_batch = self.emb(sen_batch)  # 8*40*100
        sen_batch = sen_batch.view(batch_size, 1, self.max_len, self.input_dim)  # 8*1*40*100
        sen_batch = self.conv1(sen_batch)  # 8*16*38*1
        sen_batch = F.relu(sen_batch)  # 8*16*38*1
        sen_batch = sen_batch.view(batch_size, 16, self.max_len - 3 + 1)  # 8*16*38
        sen_batch = F.max_pool2d(sen_batch, (1, self.max_len - 3 + 1))  # 8*16*1
        sen_batch = sen_batch.view(batch_size, 16)  # 8*16

        sen_batch = self.hidden1(sen_batch)  # 8*hidden_dim
        # sen_batch = F.relu(sen_batch)
        sen_batch = self.out(sen_batch)  # 8*2
        output = F.softmax(sen_batch)  # 8*2
        return output
