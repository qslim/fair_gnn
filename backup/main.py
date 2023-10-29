import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.specformer import Specformer
from data.fairgraph_dataset import POKEC, NBA
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


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.cuda)

    # Load the dataset and split
    if args.dataset == 'nba':
        dataset = NBA()
    elif args.dataset == 'pokec_z':
        dataset = POKEC(dataset_sample='pokec_z')
    elif args.dataset == 'pokec_n':
        dataset = POKEC(dataset_sample='pokec_n')
    else:
        raise ValueError('Unknown dataset!')
    adj, x, labels, idx_train, idx_val, idx_test, sens = dataset.adj, dataset.features, dataset.labels, dataset.idx_train, dataset.idx_val, dataset.idx_test, dataset.sens

    # feature_normalize
    # x = np.array(x)
    # rowsum = x.sum(axis=1, keepdims=True)
    # rowsum = np.clip(rowsum, 1, 1e10)
    # x = x / rowsum
    # x = torch.FloatTensor(x)

    e, u = [], []
    deg = np.array(adj.sum(axis=0)).flatten()
    for eps in config['eps']:
        print("Start building e, u with {}...".format(eps), end='')
        # build graph matrix
        D_ = sp.sparse.diags(deg ** eps)
        A_ = D_.dot(adj.dot(D_))
        # L_ = sp.sparse.eye(adj.shape[0]) - A_

        # eigendecomposition
        _e, _u = sp.sparse.linalg.eigsh(A_, which='LM', k=config['eigk'], tol=1e-5)
        e.append(torch.FloatTensor(_e))
        u.append(torch.FloatTensor(_u))
        print("Done.")
    e, u = torch.cat(e, dim=0).cuda(), torch.cat(u, dim=1).cuda()

    net = Specformer(1, x.size(1), config['nlayer'], config['hidden_dim'], config['num_heads'], config['tran_dropout'],
                     config['feat_dropout'], config['prop_dropout'], config['norm'], num_eigen=e.shape[0]).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net))

    best_acc = 0.0
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output = net(e, u, x)
        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        net.eval()
        output = net(e, u, x)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        parity_val, equality_val = fair_metric(output, idx_val, labels, sens)
        parity_test, equality_test = fair_metric(output, idx_test, labels, sens)

        acc_val, acc_test, parity_val, equality_val, parity_test, equality_test = acc_val * 100.0, acc_test * 100.0, parity_val * 100.0, equality_val * 100.0, parity_test * 100.0, equality_test * 100.0

        if acc_val > best_acc:
            best_epoch = epoch
            best_acc = acc_val
            best_test = acc_test
            best_dp = parity_val
            best_dp_test = parity_test
            best_eo = equality_val
            best_eo_test = equality_test

        print("Epoch {}:".format(epoch),
              "acc_test= {:.4f}".format(acc_test.item()),
              "acc_val: {:.4f}".format(acc_val.item()),
              "dp_val: {:.4f}".format(parity_val),
              "dp_test: {:.4f}".format(parity_test),
              "eo_val: {:.4f}".format(equality_val),
              "eo_test: {:.4f}".format(equality_test),
              "best_acc: {}/{:.4f}".format(best_epoch, best_test))
    print("Test results:",
          "acc_test= {:.4f}".format(best_test.item()),
          "acc_val: {:.4f}".format(best_acc.item()),
          "dp_val: {:.4f}".format(best_dp),
          "dp_test: {:.4f}".format(best_dp_test),
          "eo_val: {:.4f}".format(best_eo),
          "eo_test: {:.4f}".format(best_eo_test))
    return best_test.item(), best_acc.item(), best_dp, best_dp_test, best_eo, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='pokec_z')
    args = parser.parse_args()

    config = yaml.load(open('../config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    test, val, dp, dp_test, eo, eo_test = [], [], [], [], [], []
    for seed in args.seeds:
        args.seed = seed
        _test, _val, _dp, _dp_test, _eo, _eo_test = main_worker(args, config)
        test.append(_test)
        val.append(_val)
        dp.append(_dp)
        dp_test.append(_dp_test)
        eo.append(_eo)
        eo_test.append(_eo_test)
        
    test = np.array(test, dtype=float)
    val = np.array(val, dtype=float)
    dp = np.array(dp, dtype=float)
    dp_test = np.array(dp_test, dtype=float)
    eo = np.array(eo, dtype=float)
    eo_test = np.array(eo_test, dtype=float)
    print("Mean over {} run:".format(len(args.seeds)),
          "acc_test= {:.4f}_{:.4f}".format(np.mean(test), np.std(test)),
          "acc_val: {:.4f}_{:.4f}".format(np.mean(val), np.std(val)),
          "dp_val: {:.4f}_{:.4f}".format(np.mean(dp), np.std(dp)),
          "dp_test: {:.4f}_{:.4f}".format(np.mean(dp_test), np.std(dp_test)),
          "eo_val: {:.4f}_{:.4f}".format(np.mean(eo), np.std(eo)),
          "eo_test: {:.4f}_{:.4f}".format(np.mean(eo_test), np.std(eo_test)))

