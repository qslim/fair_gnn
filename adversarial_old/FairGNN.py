import torch.nn as nn
# from adversarial_old.eigen_gnn import EigenGNN
from adversarial.specformer import Specformer
import torch
import torch.nn.functional as F


class FairGNN(nn.Module):

    def __init__(self, nfeat, config):
        super(FairGNN,self).__init__()

        self.adver = config['adver']
        self.temp = config['temp']
        nhid = config['signal_dim']
        self.GNN = Specformer(nfeat=nfeat, nclass=1, config=config)

        # self.feat_dp2 = nn.Dropout(config['feat_dropout'])
        self.classifier = nn.Sequential(
            # nn.Linear(nhid, nhid),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            nn.Linear(nhid, 1))
        self.adv = nn.Sequential(
            nn.Linear(nhid, nhid),
            # nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1))

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = config['lr'], weight_decay = config['weight_decay'])
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def evaluate(self,e,u,x):
        z = self.GNN(e,u,x)
        # z = self.feat_dp2(z)
        y = self.classifier(z)
        return y
    
    def optimize(self,e,u,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        # self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        h = self.GNN(e,u,x)
        # h = self.feat_dp2(h)
        y = self.classifier(h)

        s_g = self.adv(h)

        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        # self.group_confusion = torch.var(s_g.squeeze())
        # self.group_confusion = torch.var(F.logsigmoid(s_g.squeeze() * self.temp))
        self.group_confusion = torch.std(F.logsigmoid(s_g.squeeze() * self.temp))
        # self.group_confusion = -torch.mean(0.5 * F.logsigmoid(s_g.squeeze()) + 0.5 * F.logsigmoid(-s_g.squeeze()))
        # print('Group Confusion:', self.group_confusion.item())

        self.G_loss = self.cls_loss + self.adver * self.group_confusion
        # self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        # self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g[idx_sens_train],sens[idx_sens_train].unsqueeze(1).float())
        self.A_loss.backward()
        self.optimizer_A.step()


