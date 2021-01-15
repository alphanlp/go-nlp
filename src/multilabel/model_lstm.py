# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class LSTMMultiLabler(nn.Module):
    """基于LSTM的多标签识别"""

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 use_bidirectional=False,
                 dropout=0.0):
        super(LSTMMultiLabler, self).__init__()

        self.n_layers = n_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=use_bidirectional,
                           dropout=dropout)

        if use_bidirectional:
            self.fcs = nn.ModuleList([nn.Linear(hidden_size * 2, output_size) for i in range(output_size)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(hidden_size, output_size) for i in range(output_size)])

    def forward(self, x):
        embedded = self.embedding(x)
        # output, (h_n, c_n)
        output, (h_n, c_n) = self.rnn(embedded)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.reshape(self.n_layers, 2 if self.use_bidirectional else 1, -1, self.hidden_size)
        h_n = h_n[-1, :, :, :]  # 最后一层
        num_direction, B, E = h_n.size()
        hidden_list = []
        for i in range(num_direction):
            hidden_list.append(h_n[i, :, :])
        h_n = torch.cat(hidden_list, dim=1)

        final_outputs = []
        for fc in self.fcs:
            final_output = fc(h_n)
            final_outputs.append(final_output)
        return final_outputs
