import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from specformer import Specformer
from data.fairgraph_dataset2 import POKEC, NBA
import scipy as sp
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.cuda)

    net_sens = Specformer(1,
                          x.size(1),
                          config['nlayer'],
                          config['hidden_dim'],
                          config['decorrela_dim'],
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
        output_sens, signal_sens = net_sens(e, u, x)
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

    # signal_sens = output_sens.detach()
    signal_sens = signal_sens.detach()
    # print(signal_sens)
    # signal_sens = torch.sigmoid(output)

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
    signal_sens = signal_sens.transpose(1, 0)
    # signal_sens = torch.sigmoid(signal_sens)
    # signal_sens = torch.abs(signal_sens)
    # signal_sens = signal_sens - signal_sens.mean(dim=1, keepdim=True)
    _signal_sens_norm = signal_sens.norm(dim=1, keepdim=True)
    _signal_sens_normed = signal_sens / torch.where(_signal_sens_norm > 1e-8, _signal_sens_norm, 1e-8)
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output, signal = net(e, u, x)

        signal = signal.transpose(1, 0)
        # signal = torch.sigmoid(signal)
        # signal = torch.abs(signal)
        # signal = signal - signal.mean(dim=1, keepdim=True)
        _signal_norm = signal.norm(dim=1, keepdim=True)
        _signal_normed = signal / torch.where(_signal_norm > 1e-8, _signal_norm, 1e-8)
        cosine = (_signal_sens_normed.unsqueeze(1) * _signal_normed.unsqueeze(0)).sum(2).abs().mean()
        # print(cosine.item())

        # cosine = torch.tensor(0.0)
        # for i in range(signal_sens.shape[0]):
        #     _cosine = F.cosine_similarity(signal_sens[i].repeat(signal.shape[0], 1), signal).abs().mean(0)
        #     cosine = cosine + _cosine
        # cosine = cosine / (signal_sens.shape[0] * 1.0)
        # print(cosine.item())

        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        loss = loss + config['decorrelation'] * cosine
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
    parser.add_argument('--dataset', default='nba')
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    

    # Load the dataset and split
    if args.dataset == 'pokec_z':
        dataset = POKEC(dataset_sample='pokec_z')
    elif args.dataset == 'pokec_n':
        dataset = POKEC(dataset_sample='pokec_n')
    elif args.dataset == 'nba':
        dataset = POKEC(data_path='https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/nba/',
                        root='./dataset/nba',
                        dataset_sample='nba',
                        sens_attr='country',
                        predict_attr='SALARY',
                        label_number=100,
                        sens_number=50,
                        test_idx=True)
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
        if False:
            _e, _u = np.linalg.eigh(A_.todense())
            _e, _u = _e[-256:], _u[:, -256:]
        else:
            _e, _u = sp.sparse.linalg.eigsh(A_, which='LM', k=config['eigk'], tol=1e-5)
        e.append(torch.FloatTensor(_e))
        u.append(torch.FloatTensor(_u))
        print("Done.")
    e, u = torch.cat(e, dim=0).cuda(), torch.cat(u, dim=1).cuda()
    # e, u = torch.stack(e, dim=0).cuda(), torch.stack(u, dim=0).cuda()



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

