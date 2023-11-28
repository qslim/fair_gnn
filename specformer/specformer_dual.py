import torch.nn as nn
from specformer import Specformer


class Specformer_wrapper(nn.Module):
    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, signal_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0):
        super(Specformer_wrapper, self).__init__()

        self.specformer_s = Specformer(1,
                              nfeat,
                              nlayer,
                              hidden_dim,
                              signal_dim,
                              nheads,
                              tran_dropout,
                              feat_dropout,
                              prop_dropout)

        self.specformer_y = Specformer(1,
                         nfeat,
                         nlayer,
                         hidden_dim,
                         hidden_dim,
                         nheads,
                         tran_dropout,
                         feat_dropout,
                         prop_dropout)

    def forward(self, e, u, x):
        pred_s, _ = self.specformer_s(e, u, x)
        pred_y, _ = self.specformer_y(e, u, x)

        return pred_y, pred_s



