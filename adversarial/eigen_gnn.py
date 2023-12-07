import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


# class FeedForwardNetwork(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(FeedForwardNetwork, self).__init__()
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             # nn.BatchNorm1d(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.LayerNorm(hidden_dim),
#             # nn.GELU(),
#         )
#
#     def forward(self, x):
#         x = self.ffn(x)
#         return x


class SpecLayer(nn.Module):

    def __init__(self, hidden_dim, signal_dim, prop_dropout=0.0):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, signal_dim),
            nn.LayerNorm(signal_dim),
            # nn.BatchNorm1d(signal_dim),
            nn.GELU()
            # nn.ELU()
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.prop_dropout(x)
        x = self.ffn(x)
        return x


class Filter(nn.Module):
    def __init__(self, hidden_dim=128, nheads=1,
                 tran_dropout=0.0):
        super(Filter, self).__init__()
        num_basis = hidden_dim

        # self.ffn_dropout = nn.Dropout(tran_dropout)
        # self.ffn = FeedForwardNetwork(num_basis, hidden_dim, hidden_dim)
        self.decoder_dropout = nn.Dropout(tran_dropout)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, e):
        eig = e
        # eig = self.ffn_dropout(eig)
        # eig = self.ffn(eig)
        eig = self.decoder_dropout(eig)
        eig = self.decoder(eig)  # [N, m]
        return eig


class EigenGNN(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, signal_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0):
        super(EigenGNN, self).__init__()

        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        # self.classify = nn.Linear(signal_dim, nclass)

        self.filter = Filter(hidden_dim=hidden_dim, nheads=nheads, tran_dropout=tran_dropout)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        layers = [SpecLayer(hidden_dim, hidden_dim, prop_dropout) for i in range(nlayer - 1)]
        layers.append(SpecLayer(hidden_dim, signal_dim, prop_dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, e, u, x):
        ut = u.permute(1, 0)
        h = self.feat_dp1(x)
        h = self.linear_encoder(h)

        filter = self.filter(e)

        for conv in self.layers:
            utx = ut @ h
            y = u @ (filter * utx)
            h = h + y
            h = conv(h)

        h = self.feat_dp2(h)
        pred = h

        return pred


