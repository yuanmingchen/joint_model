import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_CNN(nn.Module):
    def __init__(self, embeddings, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters,
                 num_classes=2, dropout=0.5, num_hidden=100, num_layers=1):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_filters = num_filters
        self.num_hidden = num_hidden

        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_size,
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.cnns = []

        for filter_size in filter_sizes:
            self.cnns.append(nn.Conv2d(in_channels=1, out_channels=num_filters,
                                       kernel_size=[filter_size, embedding_size],
                                       padding=0, stride=1).cuda())

        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=num_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False).cuda()  # 不是双向的,即不是BLSTM

        self.out = nn.Linear(in_features=96, out_features=num_classes)

    def _fetch(self, sen_batch, sen_length):
        batch_size = sen_batch.size(0)
        sen_batch = sen_batch.view(batch_size * self.sequence_length, self.num_hidden)
        index = torch.LongTensor(range(batch_size)).cuda() * self.sequence_length
        index = index + sen_length - 1
        sen_batch = torch.index_select(input=sen_batch, dim=0, index=index)
        sen_batch = sen_batch.view(batch_size, self.num_hidden)
        return sen_batch

    def forward(self, sen_batch, sen_length):
        batch_size = len(sen_batch)
        sen_batch = self.emb(sen_batch)  # 128*40*100

        sen_batch, _ = self.lstm(sen_batch)  # 128*40*100
        sen_batch = sen_batch.contiguous().view(batch_size, 1 , -1, self.num_hidden)  # (batch, sen_len, hid)
        # sen_batch = self._fetch(sen_batch, sen_length) # 128*40
        # sen_batch = sen_batch.permute([1, 0, 2])
        # sen_batch = sen_batch[len(sen_batch)-1]
        # sen_batch = sen_batch.view(batch_size, 1, self.sequence_length, self.embedding_size)  # 8*1*40*100
        cnn_outputs = []
        for i, cnn in enumerate(self.cnns):
            cnn_output = cnn(sen_batch)  # torch.Size([8, 32, 38])
            cnn_output = F.relu(cnn_output)  # torch.Size([8, 32, 38])
            cnn_output = cnn_output.view(batch_size, self.num_filters,
                                         self.sequence_length - self.filter_sizes[i] + 1)  # torch.Size([8, 32, 38])
            pool = F.max_pool2d(cnn_output, (1, self.sequence_length - self.filter_sizes[i] + 1),
                                stride=1)  # torch.Size([8, 32, 1])
            # pool = pool.view(-1, self.num_filters)
            # print(cnn_output.size(), pool.size())
            cnn_outputs.append(pool)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = torch.cat(cnn_outputs, 1)
        sen_batch = h_pool.view(-1, num_filters_total)  # 128*96*1
        output = self.out(sen_batch)
        output = F.softmax(output)
        # print(sen_batch.size(), output.size(), output)
        return output
