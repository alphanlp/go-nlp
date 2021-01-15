# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionClassfier(nn.Module):
    """基于LSTM Attention的分类器
    Implement paper: Attention-Based Bidirectional LongShort-Term Memory Networks for Relation Classiﬁcation
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 use_bidirectional=False,
                 dropout=0.0):
        super(LSTMAttentionClassfier, self).__init__()

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
            self.attn = Attention(hidden_size * 2)  # attention层
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.attn = Attention(hidden_size)  # attention层
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        # output, (h_n, c_n)
        # output [batch, seq_len, num_directions * embedding]
        output, (h_n, c_n) = self.rnn(embedded)
        attn_scores = self.attn(output)  # [batch, seq_len, 1]
        output = torch.sum(torch.mul(output, attn_scores), dim=1)
        output = torch.tanh(output)
        final_output = self.fc(output)
        return final_output


class Attention(nn.Module):
    def __init__(self, v_size):
        super(Attention, self).__init__()
        self.v_size = v_size
        self.v = nn.Parameter(torch.randn(v_size))

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        encoder_outputs = torch.tanh(encoder_outputs)
        attn_scores = torch.matmul(encoder_outputs.reshape(-1, self.v_size), self.v)
        return F.softmax(attn_scores.view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)
