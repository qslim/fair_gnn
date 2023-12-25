import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
import torch as th
import dgl.function as fn
from dgl._ffi.base import DGLError

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x):
        h = self.body(g,x)
        x = self.fc(h)
        return x, h

# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x    


class GCN_wrapper(nn.Module):
    def __init__(self, nfeat, hidden_dim=128, feat_dropout=0.0):
        super(GCN_wrapper, self).__init__()

        self.gnn_s = GCN(nfeat=nfeat,
                         nhid=hidden_dim,
                         nclass=1,
                         dropout=feat_dropout)

        self.gnn_y = GCN(nfeat=nfeat,
                         nhid=hidden_dim,
                         nclass=1,
                         dropout=feat_dropout)

    def forward(self, g, x):
        pred_s, _ = self.gnn_s(g, x)
        pred_y, _ = self.gnn_y(g, x)

        return pred_y, pred_s


class GAT_body(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_body, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads,
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads, num_hidden, heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads, num_hidden, heads,
            feat_drop, attn_drop, negative_slope, residual, None))
    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits
class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()

        self.body = GAT_body(num_layers, in_dim, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual)
        self.fc = nn.Linear(num_hidden,num_classes)
    def forward(self, g, inputs):

        logits = self.body(g,inputs)
        x = self.fc(logits)

        return x, logits


class GAT_wrapper(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_wrapper, self).__init__()

        self.gnn_s = GAT(num_layers=num_layers,
                         in_dim=in_dim,
                         num_hidden=num_hidden,
                         num_classes=1,
                         heads=heads,
                         feat_drop=feat_drop,
                         attn_drop=attn_drop,
                         negative_slope=negative_slope,
                         residual=residual)

        self.gnn_y = GAT(num_layers=num_layers,
                         in_dim=in_dim,
                         num_hidden=num_hidden,
                         num_classes=1,
                         heads=heads,
                         feat_drop=feat_drop,
                         attn_drop=attn_drop,
                         negative_slope=negative_slope,
                         residual=residual)

    def forward(self, g, x):
        pred_s, _ = self.gnn_s(g, x)
        pred_y, _ = self.gnn_y(g, x)

        return pred_y, pred_s



class SGConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(SGConv, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = deep_graph_library.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            return self.fc(feat), feat


class SGC_wrapper(nn.Module):
    def __init__(self, nfeat, num_layers=2):
        super(SGC_wrapper, self).__init__()

        self.gnn_s = SGConv(in_feats=nfeat,
                            out_feats=1,
                            k=num_layers,
                            cached=True,
                            bias=True)

        self.gnn_y = SGConv(in_feats=nfeat,
                            out_feats=1,
                            k=num_layers,
                            cached=True,
                            bias=True)

    def forward(self, g, x):
        pred_s, _ = self.gnn_s(g, x)
        pred_y, _ = self.gnn_y(g, x)

        return pred_y, pred_s
