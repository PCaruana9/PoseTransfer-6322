import torch
import torch.nn as nn
import torch.nn.functional as tfun
import numpy as np

# Peter Caruana
# York University, Toronto Canada
# EECS 6322, Winter 2021
# caruana9@my.yorku.ca

# implementations of the SPatially Adaptive Instance Normalization
# and SPatially Adaptive Batch Instance Normalization

# HELP NOTES:
#
# convolutional layer:  self.conv = torch.nn.conv1d(in_channels, out_channels, kernel_size)
#
#
#
#


class PoseFeatureExtractor(nn.Module):

    def __init__(self):
        super(PoseFeatureExtractor, self).__init__()

    def forward(self, x, identity):
        pass

class Decoder(nn.module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x, identity):
        pass

class SPAdaIN(nn.Module):

    def __init__(self, I_norm):
        super(SPAdaIN, self).__init__()

    def forward(self, x):
        pass

class SPAdaIN_ResBlock(nn.Module):

    def __init__(self):
        super(SPAdaIN_ResBlock, self).__init__()

    def forward(self, x):
        pass

class SPAdaBIN(nn.Module):

    def __init__(self, BI_norm):
        super(SPAdaBIN, self).__init__()

    def forward(self, x):
        pass

class SPAdaBIN_ResBlock(nn.Module):

    def __init__(self):
       super(SPAdaBIN_ResBlock, self).__init__()

    def forward(self, x):
        pass

class NeuralPoseTransfer(nn.Module):

    def __init__(self):
        super(NeuralPoseTransfer, self).__init__()
        self.encoder = PoseFeatureExtractor()
        self.decoder = Decoder()

    def forward(self, pose, identity):
        x = self.encoder(pose, identity)
        out = self.decoder(x, identity)
        
        return out