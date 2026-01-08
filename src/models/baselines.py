from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels*2, channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(channels*4, 1)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x).squeeze(1)

class CRNN(nn.Module):
    def __init__(self, cnn_ch=32, rnn_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(cnn_ch, cnn_ch*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.rnn = nn.GRU(input_size=cnn_ch*2, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, 1)

    def forward(self, x):
        # x: [B,1,F,T]
        z = self.cnn(x)  # [B,C,F',T']
        z = z.mean(dim=2)  # average freq -> [B,C,T']
        z = z.transpose(1,2)  # [B,T',C]
        out,_ = self.rnn(z)
        last = out[:,-1,:]
        return self.fc(last).squeeze(1)

class LSTMNet(nn.Module):
    def __init__(self, rnn_hidden=128):
        super().__init__()
        self.rnn = nn.LSTM(input_size=64, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, 1)

    def forward(self, x):
        # x: [B,1,F,T] -> [B,T,F]
        z = x.squeeze(1).transpose(1,2)
        out,_ = self.rnn(z)
        last = out[:,-1,:]
        return self.fc(last).squeeze(1)

class GRUNet(nn.Module):
    def __init__(self, rnn_hidden=128):
        super().__init__()
        self.rnn = nn.GRU(input_size=64, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, 1)

    def forward(self, x):
        z = x.squeeze(1).transpose(1,2)
        out,_ = self.rnn(z)
        last = out[:,-1,:]
        return self.fc(last).squeeze(1)
