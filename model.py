import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Peter Caruana
# York University, Toronto Canada
# EECS 6322, Winter 2021
# caruana9@my.yorku.ca

# implementations of the SPatially Adaptive Instance Normalization (SPAdaIN)
# and SPatially Adaptive Batch Instance Normalization (SPAdaBIN)

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
        self.decoder = Decoder(BOTTLENECK_SIZE + 3)  # + 3 for the xyz coordinates of the identity mesh

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

    def __init__(self, norm_type='Instance'):
        super(Decoder, self).__init__()

        Small = BOTTLENECK_SIZE // 4
        Medium = BOTTLENECK_SIZE // 2
        Large = BOTTLENECK_SIZE

        self.c1 = nn.Conv1d(Large, Large, 1)
        self.c2 = nn.Conv1d(Large, Medium, 1)
        self.c3 = nn.Conv1d(Medium, Small, 1)
        self.c4 = nn.Conv1d(Small, 3, 1)
        if norm_type == 'Batch':
            self.SPA_res1 = SPAdaBIN_ResBlock(Large)
            self.SPA_res2 = SPAdaBIN_ResBlock(Medium)
            self.SPA_res3 = SPAdaBIN_ResBlock(Small)
        else:  # instance
            self.SPA_res1 = SPAdaIN_ResBlock(Large)
            self.SPA_res2 = SPAdaIN_ResBlock(Medium)
            self.SPA_res3 = SPAdaIN_ResBlock(Small)

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
        S = BOTTLENECK_SIZE // 16  # Small
        M = BOTTLENECK_SIZE // 8  # Medium
        L = BOTTLENECK_SIZE  # Large

        self.c1 = torch.nn.Conv1d(3, S, 1)  # our points have 3 channels (x,y,z). Scaling is taken from PointNet setup
        self.c2 = torch.nn.Conv1d(S, M, 1)
        self.c3 = torch.nn.Conv1d(M, L, 1)

        self.norm1 = torch.nn.InstanceNorm1d(S)
        self.norm2 = torch.nn.InstanceNorm1d(M)
        self.norm3 = torch.nn.InstanceNorm1d(L)

    def forward(self, x):
        layer1 = F.relu(self.norm1(self.c1(x)))
        layer2 = F.relu(self.norm2(self.c2(layer1)))
        layer3 = F.relu(self.norm3(self.c3(layer2)))
        return layer3


class SPAdaIN(nn.Module):

    def __init__(self, bottleneck):
        super(SPAdaIN, self).__init__()
        self.norm = torch.nn.InstanceNorm1d(bottleneck)
        #  Identity will have 3 channels
        self.c_g = torch.nn.Conv1d(3, bottleneck, 1)
        self.c_b = torch.nn.Conv1d(3, bottleneck, 1)

    def forward(self, x, identity):
        instNorm = self.norm(x)
        beta = self.c_b(identity)
        gamma = self.c_g(identity)
        out = (gamma * instNorm) + beta
        return out


class SPAdaIN_ResBlock(nn.Module):

    #  iden_chan -> Number of channels in the identity, i.e. 3 (x,y,z)
    def __init__(self, bottleneck):
        super(SPAdaIN_ResBlock, self).__init__()
        self.SPA_1 = SPAdaIN(bottleneck)
        self.SPA_2 = SPAdaIN(bottleneck)
        self.SPA_3 = SPAdaIN(bottleneck)
        self.c1 = torch.nn.Conv1d(bottleneck, bottleneck, 1)
        self.c2 = torch.nn.Conv1d(bottleneck, bottleneck, 1)
        self.c3 = torch.nn.Conv1d(bottleneck, bottleneck, 1)

    def forward(self, x, identity):
        left = self.SPA_1(x, identity)
        left = F.relu(self.c1(left))
        left = self.SPA_2(left, identity)
        left = F.relu(self.c2(left))
        right = self.SPA_3(x, identity)
        right = F.relu(self.c3(right))

        out = left + right
        return out


# Instance Norm
class SPAdaBIN(nn.Module):

    def __init__(self, bottleneck):
        super(SPAdaBIN, self).__init__()
        self.norm = torch.nn.InstanceNorm1d(bottleneck)
        #  Identity will have 3 channels
        self.c_g = torch.nn.Conv1d(3, bottleneck, 1)
        self.c_b = torch.nn.Conv1d(3, bottleneck, 1)

    def forward(self, x, identity):
        instNorm = self.norm(x)
        beta = self.c_b(identity)
        gamma = self.c_g(identity)
        out = (gamma * instNorm) + beta
        return out


# Batch-instance norm
class SPAdaBIN_ResBlock(nn.Module):

    def __init__(self, bottleneck):
        super(SPAdaBIN_ResBlock, self).__init__()
        self.SPA_1 = SPAdaIN(bottleneck)
        self.SPA_2 = SPAdaIN(bottleneck)
        self.SPA_3 = SPAdaIN(bottleneck)
        self.c1 = torch.nn.Conv1d(bottleneck, bottleneck, 1)
        self.c2 = torch.nn.Conv1d(bottleneck, bottleneck, 1)
        self.c3 = torch.nn.Conv1d(bottleneck, bottleneck, 1)

    def forward(self, x, identity):
        left = self.SPA_1(x, identity)
        left = F.relu(self.c1(left))
        left = self.SPA_2(left, identity)
        left = F.relu(self.c2(left))
        right = self.SPA_3(x, identity)
        right = F.relu(self.c3(right))

        out = left + right

        return out
