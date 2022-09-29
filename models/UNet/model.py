#maintainer: stssashank6@gmail.com
#Note: In contrast to the paper I use padding in the encoder and decoder convolutions. I do not notice any major
# difference either way

import numpy as np
import torch as torch
import torch.nn as nn
class Unet(nn.module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channel_size=1)
        self.decoder = Decoder(channel_size=1)

    def forward(self, x):
        pass

class Encoder(nn.module):
    def __init__(self):
        super().__init__()


class Block(nn.module):
    def __init__(self, ks):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size,
                padding = "same")

        def forward(x):
            outp = self.relu(self.conv2(self.relu(self.cov1(x))))
            return outp


