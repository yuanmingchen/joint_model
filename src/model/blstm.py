import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BLSTM(nn.Module):
    def __init__(self, embeddings, input_dim=100, hidden_dim=50, num_layers=1, output_dim=2, max_len=40, dropout=0.5):
        super(BLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.emb = nn.Embedding(
            num_embeddings=embeddings.size(0),
            embedding_dim=embeddings.size(1),
            padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # 是双向的,即是BLSTM
        )
        self.out = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=output_dim
        )

    def forward(self, sen_batch, sen_length):
        batch_size = len(sen_batch)
        sen_batch = self.emb(sen_batch)
        '''
        这个地方一定要注意，lstm返回的结果是多个结果，即有一个tuple，所以务必要写上“, _”，否则sen_batch是一个tuple，而非输出结果
        LSTM 输出 output, (h_n, c_n)
        output(batch, seq_len, hidden_size * num_directions): 保存RNN最后一层的输出的Tensor。
        h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
        c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的细胞状态。
       '''
        sen_batch, _ = self.lstm(sen_batch)
        sen_batch = sen_batch.contiguous().view(batch_size, -1, self.hidden_dim * 2)  # (batch, sen_len, 2*hid)
        sen_batch = self._fetch(sen_batch, sen_length)
        output = self.out(sen_batch)
        output = F.softmax(output)
        return output

    def _fetch(self, sen_batch, sen_length):
        batch_size = sen_batch.size(0)
        rnn_outs = sen_batch.view(batch_size, self.max_len, 2, -1)

        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * self.max_len, -1)

        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * self.max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * self.max_len
        batch_zeors = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + sen_length.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)

        bw_index = batch_range + batch_zeors
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs
        # sen_batch = sen_batch.view(batch_size * self.max_len, self.hidden_dim * 2)
        # index = torch.LongTensor(range(batch_size)).cuda() * self.max_len
        # index = index + sen_length - 1
        # sen_batch = torch.index_select(input=sen_batch, dim=0, index=index)
        # sen_batch = sen_batch.view(batch_size, self.hidden_dim * 2)
        # return sen_batch
