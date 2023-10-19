import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.ffn(x)
        return x


class SpecLayer(nn.Module):

    def __init__(self, hidden_dim, prop_dropout=0.0):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.prop_dropout(x)
        x = self.ffn(x)
        return x


class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none', num_eigen=-1):
        super(Specformer, self).__init__()

        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        self.layers = nn.ModuleList(
            [SpecLayer(hidden_dim, prop_dropout) for i in range(nlayer)])

    def forward(self, e, u, x):
        ut = u.permute(1, 0)
        h = self.feat_dp1(x)
        h = self.linear_encoder(h)

        eig = self.eig_encoder(e)  # [N, d]
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)
        new_e = self.decoder(eig)  # [N, m]

        for conv in self.layers:
            utx = ut @ h
            y = u @ (new_e * utx)
            h = h + y
            h = conv(h)

        h = self.feat_dp2(h)
        h = self.classify(h)
        return h

