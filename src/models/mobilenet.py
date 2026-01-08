from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v2(weights=None)
        # adapt first conv to 1-channel
        first = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                                     stride=first.stride, padding=first.padding, bias=False)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(m.last_channel, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)
