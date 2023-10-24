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
from model2 import Specformer
from fairgraph_dataset import POKEC, NBA
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
    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = dataset.adj, dataset.features, dataset.labels, dataset.idx_train, dataset.idx_val, dataset.idx_test, dataset.sens, dataset.idx_sens_train

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

    net_sens = Specformer(1,
                          x.size(1),
                          config['nlayer'],
                          config['hidden_dim'],
                          config['orthog_dim'],
                          config['num_heads'],
                          config['tran_dropout'],
                          config['feat_dropout'],
                          config['prop_dropout'],
                          config['norm']).cuda()
    net_sens.apply(init_params)
    optimizer = torch.optim.Adam(net_sens.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net_sens))

    best_acc = 0.0
    for epoch in range(config['epoch']):
        net_sens.train()
        optimizer.zero_grad()
        output_sens, emb_sens = net_sens(e, u, x)
        loss = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
        acc_train = accuracy(output_sens[idx_sens_train], sens[idx_sens_train])
        loss.backward()
        optimizer.step()

        net_sens.eval()
        output_sens, _ = net_sens(e, u, x)
        acc_val = accuracy(output_sens[idx_val], sens[idx_val])
        acc_test = accuracy(output_sens[idx_test], sens[idx_test])

        acc_val, acc_test = acc_val * 100.0, acc_test * 100.0

        if acc_val > best_acc:
            best_epoch = epoch
            best_acc = acc_val
            best_test = acc_test

        # print("Stage 1 Epoch {}:".format(epoch),
        #       "acc_test= {:.4f}".format(acc_test.item()),
        #       "acc_val: {:.4f}".format(acc_val.item()),
        #       "best_acc: {}/{:.4f}".format(best_epoch, best_test))
    print("Test results:",
          "acc_test= {:.4f}".format(best_test.item()),
          "acc_val: {:.4f}".format(best_acc.item()))

    # sens_embedding = output_sens.detach()
    sens_embedding = emb_sens.detach()
    # print(sens_embedding)
    # sens_embedding = torch.sigmoid(output)

    net = Specformer(1,
                     x.size(1),
                     config['nlayer'],
                     config['hidden_dim'],
                     config['hidden_dim'],
                     config['num_heads'],
                     config['tran_dropout'],
                     config['feat_dropout'],
                     config['prop_dropout'],
                     config['norm']).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net))

    best_acc = 0.0
    # sens_embedding = sens_embedding.squeeze().repeat(config['hidden_dim'], 1)
    sens_embedding = sens_embedding.transpose(1, 0)
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output, emb = net(e, u, x)
        cosine = torch.tensor(0.0)
        emb_t = emb.transpose(1, 0)
        for i in range(sens_embedding.shape[0]):
            cosine_i = F.cosine_similarity(sens_embedding[i].repeat(emb_t.shape[0], 1), emb_t).abs().mean(0)
            # cosine.append(cosine_i)
            cosine = cosine + cosine_i
        # cosine = torch.cat(cosine, dim=0).mean(0)
        cosine = cosine / (sens_embedding.shape[0] * 1.0)
        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        loss = loss + config['orthogonal'] * cosine
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        net.eval()
        output, _ = net(e, u, x)
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

        print("Stage 2 Epoch {}:".format(epoch),
              "loss: {:.4f}".format(loss.item()),
              "cosine: {:.4f}".format(cosine.item()),
              "acc_test: {:.4f}".format(acc_test.item()),
              "acc_val: {:.4f}".format(acc_val.item()),
              "dp_val: {:.4f}".format(parity_val),
              "eo_val: {:.4f}".format(equality_val),
              "[dp_test: {:.4f}".format(parity_test),
              "eo_test: {:.4f}]".format(equality_test),
              "best_acc: {}/{:.4f}".format(best_epoch, best_test))
    print("Test results:",
          "acc_test= {:.4f}".format(best_test.item()),
          "acc_val: {:.4f}".format(best_acc.item()),
          "dp_val: {:.4f}".format(best_dp),
          "eo_val: {:.4f}".format(best_eo),
          "[dp_test: {:.4f}".format(best_dp_test),
          "eo_test: {:.4f}]".format(best_eo_test))
    return best_test.item(), best_acc.item(), best_dp, best_dp_test, best_eo, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='pokec_z')
    args = parser.parse_args()

    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
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
          "eo_val: {:.4f}_{:.4f}".format(np.mean(eo), np.std(eo)),
          "[dp_test: {:.4f}_{:.4f}".format(np.mean(dp_test), np.std(dp_test)),
          "eo_test: {:.4f}_{:.4f}]".format(np.mean(eo_test), np.std(eo_test)))

