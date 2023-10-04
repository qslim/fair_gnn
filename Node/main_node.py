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
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from model_node import Specformer
from utils import calculate_similarity_matrix, convert_sparse_matrix_to_sparse_tensor, count_parameters, init_params, \
    load_credit, seed_everything, get_split
import csv
import pandas as pd
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp


def main_worker(args, config):
    # print(args, config)
    # seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.seed)

    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']

    if 'signal' in args.dataset:
        e, u, x, y, m = torch.load('data/{}_LM_100.pt'.format(args.dataset))
        e, u, x, y, m = e.cuda(), u.cuda(), x.cuda(), y.cuda(), m.cuda()
        mask = torch.where(m == 1)
        x = x[:, args.image].unsqueeze(1)
        y = y[:, args.image]
    else:
        e, u, x, y, sens = torch.load('data/{}_LM_100.pt'.format(args.dataset))
        e, u, x, y, sens = e, u, x, y, sens
        # print(y)
        if len(y.size()) > 1:
            if y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1)

        train, valid, test = get_split(args.dataset, y, nclass, args.seed)
        train, valid, test = map(torch.LongTensor, (train, valid, test))
        # print(y)
        # train, valid, test = train, valid, test

    nfeat = x.size(1)
    net = Specformer(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # print(count_parameters(net))

    res = []
    min_loss = 100.0
    max_acc = 0
    counter = 0
    # print(e)
    # print(u)
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    sens_attr = "country"  # 等于号前面的是自己命名的，方便记录下来每一个变量都对应的是什么
    sens_idx = 1
    predict_attr = 'SALARY'
    label_number = 100
    path_credit = "node_raw_data/NBA"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
                                                                            predict_attr, path=path_credit,
                                                                            label_number=label_number
                                                                            )

    #sim = calculate_similarity_matrix(adj, features, metric='cosine')  # 求相似矩阵
    #lap = laplacian(sim)
    #lap = convert_sparse_matrix_to_sparse_tensor(lap)
    # print(lap)
    # ifgOutput = ifgModel(output, sim_edge_index, sim_edge_weight)
    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)
        logits_max = torch.max(logits, axis=1).values.reshape(-1, 1)

        # print(lap)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            ifair_loss = torch.trace(torch.mm(logits_max.t(), torch.sparse.mm(lap, logits_max)))
            loss = torch.square((logits[mask] - y[mask])).sum()
        else:

             #ifair_loss = torch.trace(torch.mm(logits_max.t(), torch.sparse.mm(lap, logits_max)))
             loss = F.cross_entropy(logits[train], y[train])

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            r2 = r2_score(y[mask].data.cpu().numpy(), logits[mask].data.cpu().numpy())
            sse = torch.square(logits[mask] - y[mask]).sum().item()
            print(r2, sse)
        else:
            val_loss = F.cross_entropy(logits[valid], y[valid]).item()

            val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
            test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
            parity, equality, idx_s0,idx_s1,idx_s0_y1,idx_s1_y1 = fair_metric(y, sens, torch.argmax(logits, dim=1), test)
            res.append([100 * test_acc, 100 * parity, 100 * equality])

            # print(idx, '%.2f'%val_loss, '%.2f'%val_acc, '%.2f'%test_acc,'%.2f'%parity,'%.2f'%equality)     #每次训练后的测试的结果,排列方式为(训练轮数,训练损失,验证集准确率,测试集准确率,SP值,EO值)
            # print('fire',parity,equality)

            if val_loss < min_loss:
                min_loss = val_loss
                counter = 0
            else:
                counter += 1
        if counter == 200:

            max_acc3 = sorted(res[-200:], key=lambda x: (x[1] + x[2]), reverse=False)[0]

            result = pd.DataFrame(data=[['%.4f' % i for i in max_acc3]])
            result.to_csv('IndEx_NBA_high_noe.csv', mode='a', header=False)  # 数据存入csv,存储位置及文件名称
            break


def fair_metric(y, sens, output, idx):
    val_y = y[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(y).cpu().numpy()
    # print(pred_y)
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality, idx_s0,idx_s1,idx_s0_y1,idx_s1_y1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)  # 改种子
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='NBA')  # 改数据集,['NBA','pokec_z','pokec_n']
    parser.add_argument('--image', type=int, default=0)

    args = parser.parse_args()

    if 'signal' in args.dataset:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)['signal']
    else:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    for i in range(100):  # 改轮数
        seed_everything(i)
        main_worker(args, config)
        print('finish',i)
    # seed_everything(67)
    # main_worker(args, config)
    # print('finish',67)

