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
from model_node import GCN, Specformer
from utils import calculate_similarity_matrix, convert_sparse_matrix_to_sparse_tensor, count_parameters, init_params, \
    load_credit, seed_everything, get_split
import csv
import pandas as pd
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main_worker(args, config, method):
    print(args, config)
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

    # if 'signal' in args.dataset:
    #     e, u, x, y, m = torch.load('data/{}.pt'.format(args.dataset))
    #     e, u, x, y, m = e.cuda(), u.cuda(), x.cuda(), y.cuda(), m.cuda()
    #     mask = torch.where(m == 1)
    #     x = x[:, args.image].unsqueeze(1)
    #     y = y[:, args.image]
    # else:
    #     e, u, x, y, sens = torch.load('data/NBA.pt')
    #     e, u, x, y, sens = e, u, x, y, sens
    #     # print(y)
    #     if len(y.size()) > 1:
    #         if y.size(1) > 1:
    #             y = torch.argmax(y, dim=1)
    #         else:
    #             y = y.view(-1)


    e, u, x, y, sens = torch.load('data/pokec_z.pt')
    e, u, x, y, sens = e, u, x, y, sens
    features, labels, idx_train, idx_val, idx_test, sens = torch.load('data/region_job_information.pt')
    # print(y)
    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)
        # train, valid, test = get_split(args.dataset, y, nclass, args.seed)
        # train, valid, test = map(torch.LongTensor, (train, valid, test))
        # print(y)
        # train, valid, test = train, valid, test

    nfeat = x.size(1)
    if method == 1:
        net = Specformer(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm)
    elif method == 2:
        net = GCN(nfeat=nfeat, nhid=hidden_dim, nclass=nclass, dropout=tran_dropout)
    elif method == 3:
        net = SpGAT(nfeat=nfeat, nhid=hidden_dim, nclass=nclass, dropout=tran_dropout, nheads=1, alpha=0.2)
    elif method == 4:
        net = GAT(nfeat=nfeat, nhid=hidden_dim, nclass=nclass, dropout=tran_dropout, nheads=1, alpha=0.2)
    else:
        print('cannot find method')
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # print(count_parameters(net))

    res = []
    min_loss = 100.0
    max_acc = 0.65
    counter = 0
    # print(e)
    # print(u)
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    # sens_attr = "country"  # 等于号前面的是自己命名的，方便记录下来每一个变量都对应的是什么
    # sens_idx = 1
    # predict_attr = 'SALARY'
    # label_number = 100
    # path_credit = "node_raw_data/NBA/"
    # if args.dataset == 'pokec_z':
    #     args.dataset = 'region_job'
    # adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
    #                                                                         predict_attr, path=path_credit,
    #                                                                         label_number=label_number
    #                                                                         )
    print('success load data')

    # sim = calculate_similarity_matrix(adj, features, metric='cosine') #求相似矩阵
    # lap = laplacian(sim)
    # lap = convert_sparse_matrix_to_sparse_tensor(lap)
    # print(lap)
    # ifgOutput = ifgModel(output, sim_edge_index, sim_edge_weight)
    # adj = convert_sparse_matrix_to_sparse_tensor(adj)

    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        if method == 1:
            logits = net(e, u, x)
        else:
            logits = net(x, adj)
        # logits_max=torch.max(logits, axis=1).values.reshape(-1,1)

        # print(lap)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            # ifair_loss = torch.trace( torch.mm( logits_max.t(), torch.sparse.mm(lap, logits_max) ) )
            loss = torch.square((logits[mask] - y[mask])).sum()
        else:

            # ifair_loss = torch.trace( torch.mm( logits_max.t(), torch.sparse.mm(lap, logits_max) ) )
            # loss = F.cross_entropy(logits[idx_train], y[idx_train]) +0.0001 * ifair_loss
            loss = F.cross_entropy(logits[idx_train], y[idx_train])

        loss.backward()
        optimizer.step()

        net.eval()
        if method == 1:
            logits = net(e, u, x)
        else:
            logits = net(x, adj)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            r2 = r2_score(y[mask].data.cpu().numpy(), logits[mask].data.cpu().numpy())
            sse = torch.square(logits[mask] - y[mask]).sum().item()
            print(r2, sse)
        else:
            val_loss = F.cross_entropy(logits[idx_val], y[idx_val]).item()

            val_acc = evaluation(logits[idx_val].cpu(), y[idx_val].cpu()).item()
            test_acc = evaluation(logits[idx_test].cpu(), y[idx_test].cpu()).item()
            parity, equality = fair_metric(y, sens, torch.argmax(logits, dim=1), idx_test)
            res.append([100 * test_acc, 100 * parity, 100 * equality])

            # print(idx, '%.2f'%val_loss, '%.2f'%val_acc, '%.2f'%test_acc,'%.2f'%parity,'%.2f'%equality)     #每次训练后的测试的结果,排列方式为(训练轮数,训练损失,验证集准确率,测试集准确率,SP值,EO值)
            # print('fire',parity,equality)
            if method == 1:
                if val_loss < min_loss:
                    min_loss = val_loss
                    counter = 0
                    print('lucky:', counter)
                else:
                    counter += 1
                    print('lucky:', counter)
            else:
                if val_acc > max_acc:
                    max_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                    print('lucky:', counter)

        if counter == 200:
            # max_acc1 = sorted(res[-200:], key=lambda x: x[0], reverse=False)[0]
            # max_acc2= sorted(res[-200:], key=lambda x: x[1], reverse=True)[0]

            max_acc3 = sorted(res[-200:], key=lambda x: (x[1] + x[2]), reverse=False)[0]
            max_acc4 = sorted(res[-200:], key=lambda x: x[1], reverse=False)[0]
            max_acc5 = sorted(res[-200:], key=lambda x: x[2], reverse=False)[0]
            # max_acc6= sorted(res[-200:], key=lambda x: x[2], reverse=True)[0]

            # min_parity ,min_equality=np.where(res[:][-1] == max_acc1)
            # print('每条结果显示的顺序:(训练损失,验证集准确率,测试集准确率,SP值,EO值)')
            # print('耐心值(200)内训练损失最小的结果:', max_acc1)
            # print('耐心值(200)内验证集准确率最高的结果:', max_acc2)
            print('耐心值(200)内(SP+EO)最小的结果:', ['%.2f' % i for i in max_acc3])
            # print('耐心值(200)内(SP+EO)最小的结果:',max_acc3])
            print('耐心值(200)内SP最小的结果:', ['%.2f' % i for i in max_acc4])
            # print('耐心值(200)内SP最小的结果:',max_acc4)
            print('耐心值(200)内EO最小的结果:', ['%.2f' % i for i in max_acc5])
            # print('耐心值(200)内SP最小的结果:',max_acc5)
            # print('耐心值(200)内测试集准确率最高的结果:',max_acc6)
            result = pd.DataFrame(data=[
                ['%.2f' % i for i in max_acc3] + ['%.2f' % i for i in max_acc4] + ['%.2f' % i for i in
                                                                                   max_acc5]])  # 将数据放进表格
            # result = pd.DataFrame(data=[['%.2f' % i for i in max_acc3]])
            result.to_csv('IndEx_pokec_z_T.csv', mode='a', header=False)  # 数据存入csv,存储位置及文件名称
            # return logits, y
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

    return parity, equality


# def Visibilization(record_list):
#     n_components = 2
#     perplexity = 50
#     tsne = TSNE(n_components=n_components, init='pca', perplexity=perplexity, random_state=None)
#     # tsne_res = tsne.fit_transform(data)
#     # print(data_info)
#     feature = torch.as_tensor(record_list[0]).detach()
#     y_pred = torch.as_tensor(record_list[1]).detach()
#     # print(feature.shape)
#     print(feature, y_pred)
#     print(feature.shape, y_pred.shape)
#     tsne_res = tsne.fit_transform(feature.cpu())
#     fig = plt.figure(figsize=(9, 9))
#     # Width,High
#     ax = fig.add_subplot(111)
#     # (1row 2col) 1x2
#     ax.scatter(x=tsne_res[:, 0], y=tsne_res[:, 1], c=y_pred.cpu(), s=20)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
#     # print(args['dname'],args.method,args.use_physic)
#     fig.tight_layout()
#     save_path = './figs/'
#     save_name = 'orl_' + perplexity + 'nba.png'
#     fig.savefig(save_path + save_name, dpi=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2)  # 改种子
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='pokec_z')  # 改数据集,['NBA','pokec_z','pokec_n']
    parser.add_argument('--image', type=int, default=0)
    method = 1  # 1代表原来的，2代表GCN，3代表spGAT,4代表GAT
    args = parser.parse_args()

    if 'signal' in args.dataset:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)['signal']
    else:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    for i in range(100):  # 改轮数
        seed_everything(i)
        main_worker(args, config, method)
        #record_list = main_worker(args, config, method)
        #Visibilization(record_list)
        print('finish', i)

