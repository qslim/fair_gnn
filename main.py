import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model import Specformer
from fairgraph_dataset import POKEC
import scipy as sp


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    torch.cuda.set_device(args.seed)

    # Load the dataset and split
    pokec = POKEC(dataset_sample='pokec_z')  # you may also choose 'pokec_n'
    adj, x, y, idx_train, idx_val, idx_test, sens = pokec.adj, pokec.features, pokec.labels, pokec.idx_train, pokec.idx_val, pokec.idx_test, pokec.sens

    # feature_normalize
    # x = np.array(x)
    # rowsum = x.sum(axis=1, keepdims=True)
    # rowsum = np.clip(rowsum, 1, 1e10)
    # x = x / rowsum
    # x = torch.FloatTensor(x)

    # eigendecomposition
    print("Start sp.sparse.linalg.eigsh...", end='')
    e, u = sp.sparse.linalg.eigsh(adj, which='LM', k=10)
    e = torch.FloatTensor(e).cuda()
    u = torch.FloatTensor(u).cuda()
    print("Done.")

    net = Specformer(1, x.size(1), config['nlayer'], config['hidden_dim'], config['num_heads'], config['tran_dropout'],
                     config['feat_dropout'], config['prop_dropout'], config['norm']).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net))

    best_acc = 0.0
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)
        loss = F.binary_cross_entropy_with_logits(logits[idx_train], y[idx_train].unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x)
        acc_val = accuracy(logits[idx_val], y[idx_val])
        acc_test = accuracy(logits[idx_test], y[idx_test])
        parity, equality = fair_metric(logits, idx_test, y, sens)
        # print(100 * acc_test, 100 * parity, 100 * equality)

        if acc_val > best_acc:
            best_acc = acc_val
            best_test = acc_test

        print("Epoch {}:".format(epoch),
              "acc_test= {:.6f}".format(acc_test.item()),
              "acc_val: {:.6f}".format(acc_val.item()),
              "dp_test: {:.6f}".format(parity),
              "eo_test: {:.6f}".format(equality),
              "best_acc: {:.6f}".format(best_test))


def fair_metric(output, idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='pokec_z')
    args = parser.parse_args()

    config = yaml.load(open('config_pokec.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    main_worker(args, config)

