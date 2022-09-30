#Note: In contrast to the paper I use padding in the encoder and decoder convolutions. I do not notice any major
# difference either way

import numpy as np
import torch as torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        ds = cfg.dataset
        m = cfg.model
        self.encoder = Encoder(m.encoder)
        self.decoder = Decoder(m.decoder, ds.num_classes)

    def forward(self, x):
        outp, feats = self.encoder(x)
        outp = self.decoder(outp, feats)
        return outp

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        N = len(cfg.channel_sizes)
        self.blocks = nn.ModuleList([Block(cfg.channel_sizes[i], cfg.channel_sizes[i+1], cfg.block_kernel_size) for i in range(N-1)]) 
        self.maxpool = nn.MaxPool2d(kernel_size = cfg.maxpool_kernel_size,stride=cfg.stride)

    def forward(self, x):
        feats = []
        feat = x
        for i in range(len(self.blocks)):
            feat = self.blocks[i](feat)
            feats.append(feat)
            feat = self.maxpool(feat)
        return feats[-1], feats[:-1]


class Decoder(nn.Module):
    def __init__(self, cfg, num_classes):
        super(Decoder, self).__init__()
        N = len(cfg.channel_sizes)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(cfg.channel_sizes[i], cfg.channel_sizes[i+1], kernel_size =
            cfg.upconv_kernel_size, stride = cfg.stride) for i in range(N-1)])
        self.blocks = nn.ModuleList([Block(cfg.channel_sizes[i], cfg.channel_sizes[i+1], cfg.block_kernel_size) for i in range(N-1)]) 
        self.conv = nn.Conv2d(cfg.channel_sizes[-1], num_classes, kernel_size=3,padding="same")

    def forward(self, x, feats):
        outp = x
        for i in range(len(self.blocks)):
            outp = self.upconvs[i](outp)
            outp = torch.cat([feats[-(i+1)], outp], axis=1)
            outp = self.blocks[i](outp)
        outp = self.conv(outp)
        return outp


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size,
                padding = "same")

    def forward(self, x):
        outp = self.relu(self.conv2(self.relu(self.conv1(x))))
        return outp


