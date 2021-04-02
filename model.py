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
# Concatination of tensors: torch.cat((x1, x2, ...) 1)
#
#
# Original paper used 1d convolutional kernels of size k=1

BOTTLENECK_SIZE = 1024
POINTS = 6890  # number of points in models from SMPL


class NeuralPoseTransfer(nn.Module):

    def __init__(self):
        super(NeuralPoseTransfer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(BOTTLENECK_SIZE + 3)  # +3 for the xyz coordinates of the identity mesh

    def forward(self, pose, identity):
        x = self.encoder(pose, identity)
        out = self.decoder(x, identity)

        return out.transpose(2, 1)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.pfe = PoseFeatureExtractor()

    def forward(self, x, identity):
        x_pfe = self.pfe(x)
        cat = torch.cat((x_pfe, identity), 1)
        return cat

class Decoder(nn.Module):

    def __init__(self, bottleneck_size, norm_type='Instance'):
        super(Decoder, self).__init__()
        self.c1 = nn.conv1d(bottleneck_size, bottleneck_size, 1)
        self.c2 = nn.conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.c3 = nn.conv1d(bottleneck_size, bottleneck_size//4, 1)
        self.c4 = nn.conv1d(bottleneck_size//4, 3, 1)
        if norm_type == 'Batch':
            self.SPA_res1 = SPAdaBIN_ResBlock()
            self.SPA_res2 = SPAdaBIN_ResBlock()
            self.SPA_res3 = SPAdaBIN_ResBlock()
        else: #instance
            self.SPA_res1 = SPAdaIN_ResBlock()
            self.SPA_res2 = SPAdaIN_ResBlock()
            self.SPA_res3 = SPAdaIN_ResBlock()

    def forward(self, x, identity):
        x1 = self.c1(x)
        x1 = self.SPA_res1(x1, identity)
        x1 = self.c2(x1)
        x1 = self.SPA_res2(x1, identity)
        x1 = self.c3(x1)
        x1 = self.SPA_res3(x1, identity)
        x1 = self.c4(x1)
        out = torch.nn.functional.tanh(x1)
        return out

class PoseFeatureExtractor(nn.Module):

    def __init__(self):
        super(PoseFeatureExtractor, self).__init__()
        self.c1 = torch.nn.Conv1d(3, 64, 1)
        self.c2 = nn.functional.instance_(64, 128, 1)
        self.c3 = nn.functional.instance_(128, 1024, 1)
        self.norm1 = torch.nn.InstanceNorm1d() # instance norms
        self.norm2 = torch.nn.InstanceNorm1d()
        self.norm3 = torch.nn.InstanceNorm1d()

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.norm1(x1)
        x1 = self.c2(x1)
        x1 = self.norm2(x1)
        x1 = self.c3(x1)
        out = self.norm4(x1)
        return out

class SPAdaIN(nn.Module):

    def __init__(self, I_norm):
        super(SPAdaIN, self).__init__()
        self.norm = torch.nn.InstanceNorm1d()()
        torch.nn.Conv
        self.c_x = torch.nn.Conv1d() # X
        self.c_p = torch.nn.Conv1d() # +

    def forward(self, x, identity):
        instNorm = self.norm(x)
        beta = self.c_p(identity)
        gamma = self.c_x(identity)
        out = (gamma * instNorm) + beta
        return out



class SPAdaIN_ResBlock(nn.Module):

    def __init__(self):
        super(SPAdaIN_ResBlock, self).__init__()
        self.SPA_1 = SPAdaIN()
        self.SPA_2 = SPAdaIN()
        self.SPA_3 = SPAdaIN()
        self.c1 = torch.nn.Conv1d()
        self.c2 = torch.nn.Conv1d()
        self.c3 = torch.nn.Conv1d()

    def forward(self, x, identity):
        left = self.SPA_1(x, identity)
        left = self.c1(left)
        left = self.SPA_2(left, identity)
        left = self.c2(left)
        right = self.SPA_3(x, identity)
        right = self.c3(right)

        out = left + right
        return out

# Instance Norm
class SPAdaBIN(nn.Module):

    def __init__(self, BI_norm):
        super(SPAdaBIN, self).__init__()

    def forward(self, x, identity):
        pass

# Batch-instance norm
class SPAdaBIN_ResBlock(nn.Module):

    def __init__(self):
       super(SPAdaBIN_ResBlock, self).__init__()

    def forward(self, x):
        pass

