# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.multilabel.model_lstm import LSTMMultiLabler

torch.manual_seed(1)

model = LSTMMultiLabler(vocab_size=10,
                        embedding_size=8,
                        hidden_size=16,
                        output_size=2,
                        n_layers=1,
                        use_bidirectional=True,
                        dropout=0.2)

if __name__ == '__main__':
    x = [[1, 2, 3], [2, 3, 4], [4, 5, 6], [4, 6, 5]]
    y = [[1, 0], [1, 1], [0, 1], [0, 1]]  # batch, num_label
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

            # targets  [batch, num_classes]
            inputs, targets = x_batch, y_batch
            outputs = model(inputs)

            loss = torch.FloatTensor(1, ).zero_()
            for index in range(len(outputs)):
                sub_loss = criterion(outputs[index], targets[:, index])
                loss += sub_loss

            loss.backward()
            optimizer.step()
            step += 1

            if step % 10 == 0:
                print("training step={}, epoch={}, loss={}".format(step, epoch, loss.item()))
