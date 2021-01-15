# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.classification.model_lstm import LSTMClassfier
from src.classification.model_lstm_attn import LSTMAttentionClassfier

torch.manual_seed(1)

# model = LSTMClassfier(vocab_size=10,
#                           embedding_size=8,
#                           hidden_size=16,
#                           output_size=2,
#                           n_layers=1,
#                           use_bidirectional=True,
#                           dropout=0.0)

model = LSTMAttentionClassfier(vocab_size=10,
                               embedding_size=8,
                               hidden_size=16,
                               output_size=2,
                               n_layers=2,
                               use_bidirectional=True,
                               dropout=0.2)

if __name__ == '__main__':
    x = [[1, 2, 3], [2, 3, 4], [4, 5, 6], [4, 6, 5]]
    y = [1, 1, 0, 0]
    train_data = TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))

    model.train()
    epochs = 1000
    step = 0

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        train_loader = DataLoader(train_data, batch_size=4)
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            inputs, targets = x_batch, y_batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            step += 1

            if step % 10 == 0:
                print("training step={}, epoch={}, loss={}".format(step, epoch, loss.item()))
