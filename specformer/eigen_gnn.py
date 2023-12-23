import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128, power=0.5, step=0.1):
        super(SineEncoding, self).__init__()
        # self.constant = 1.0
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)
        self.power = power
        self.step = step

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        # ee = e * self.constant
        div = torch.arange(1.0, self.hidden_dim * self.step + 1.0, self.step).to(e.device)
        pe = e.unsqueeze(1) * div
        sign = pe / pe.abs()
        spec = [e.unsqueeze(1), pe.abs().pow(self.power) * sign]
        eeig = torch.cat(spec, dim=1)

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

    def __init__(self, hidden_dim, signal_dim, prop_dropout=0.0):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, signal_dim),
            nn.LayerNorm(signal_dim),
            nn.GELU()
            # nn.ELU()
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.prop_dropout(x)
        x = self.ffn(x)
        return x


class Filter(nn.Module):
    def __init__(self, config):
        super(Filter, self).__init__()

        self.eig_encoder = SineEncoding(hidden_dim=config['hidden_dim'])
        self.decoder = nn.Linear(config['hidden_dim'], 1)

        self.mha_norm = nn.LayerNorm(config['hidden_dim'])
        self.ffn_norm = nn.LayerNorm(config['hidden_dim'])
        self.mha_dropout = nn.Dropout(config['tran_dropout'])
        self.ffn_dropout = nn.Dropout(config['tran_dropout'])
        self.mha = nn.MultiheadAttention(config['hidden_dim'], config['num_heads'], config['tran_dropout'])
        self.ffn = FeedForwardNetwork(config['hidden_dim'], config['hidden_dim'], config['hidden_dim'])

    def forward(self, e):
        eig = self.eig_encoder(e)  # [N, d]
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)
        new_e = self.decoder(eig)  # [N, m]
        return new_e


class Specformer(nn.Module):

    def __init__(self, nfeat, nclass=1, config=None):
        super(Specformer, self).__init__()

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, config['hidden_dim']),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.classify = nn.Linear(config['signal_dim'], nclass)

        self.filter = Filter(config)

        self.feat_dp1 = nn.Dropout(config['feat_dropout'])
        self.feat_dp2 = nn.Dropout(config['feat_dropout'])
        layers = [SpecLayer(config['hidden_dim'], config['hidden_dim'], config['prop_dropout']) for i in range(config['nlayer'] - 1)]
        layers.append(SpecLayer(config['hidden_dim'], config['signal_dim'], config['prop_dropout']))
        self.layers = nn.ModuleList(layers)

    def forward(self, e, u, x):
        ut = u.permute(1, 0)
        h = self.feat_dp1(x)
        h = self.feat_encoder(h)

        filter = self.filter(e)

        for conv in self.layers:
            utx = ut @ h
            y = u @ (filter * utx)
            h = h + y
            h = conv(h)

        h = self.feat_dp2(h)
        pred = self.classify(h)

        return pred


class Specformer_wrapper(nn.Module):
    def __init__(self, nfeat, config, shd_filter=False, shd_trans=False):
        super(Specformer_wrapper, self).__init__()

        self.specformer_s = Specformer(nfeat=nfeat, nclass=1, config=config)
        config['signal_dim'] = config['hidden_dim']
        self.specformer_y = Specformer(nfeat=nfeat, nclass=1, config=config)

        if shd_filter:
            print('Applying shd_filter...')
            self.specformer_s.filter = self.specformer_y.filter
        if shd_trans:
            print('Applying shd_trans...')
            self.specformer_s.layers = self.specformer_y.layers

    def forward(self, e, u, x):
        pred_s = self.specformer_s(e, u, x)
        pred_y = self.specformer_y(e, u, x)

        return pred_y, pred_s
