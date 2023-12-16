import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import linalg
import sys
sys.path.append('..')
from specformer import Specformer, Classifier
from data.Preprocessing import load_data
import scipy as sp
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric, evaluation_results, group_by_attr
from result_stat.result_append import result_append


def fit_label(H_sen0):
    print("--------------------------Stage2-------------------------")
    H_sen0 = H_sen0.detach()
    net_stage2 = Classifier(config['signal_dim'],
                             config['signal_dim'],
                             1).cuda()
    net_stage2.apply(init_params)
    optimizer = torch.optim.Adam(net_stage2.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net_stage2))
    best_acc = 0.0
    best_epoch = -1
    best_auc_roc_test = 0.0
    best_f1_s_test = 0.0
    best_test = 0.0
    best_dp_test = 1e5
    best_eo_test = 1e5
    for epoch in range(2000):
        net_stage2.train()
        optimizer.zero_grad()
        logit = net_stage2(H_sen0)
        loss = F.binary_cross_entropy_with_logits(logit[idx_train], labels[idx_train].unsqueeze(1).float())
        acc_train = accuracy(logit[idx_train], labels[idx_train]) * 100.0
        loss.backward()
        optimizer.step()
        net_stage2.eval()
        logit = net_stage2(H_sen0)
        acc_val = accuracy(logit[idx_val], labels[idx_val]) * 100.0
        acc_test = accuracy(logit[idx_test], labels[idx_test]) * 100.0
        parity_test, equality_test = fair_metric(logit, idx_test, labels, sens)
        auc_roc_test, f1_s_test, _ = evaluation_results(logit, labels, idx_test)
        parity_test, equality_test = parity_test * 100.0, equality_test * 100.0
        auc_roc_test, f1_s_test = auc_roc_test * 100.0, f1_s_test * 100.0

        # if loss_val < best_loss:
        #     best_loss = loss_val.item()
        if acc_val > best_acc:
            best_acc = acc_val.item()
            best_epoch = epoch
            best_auc_roc_test = auc_roc_test.item()
            best_f1_s_test = f1_s_test.item()
            best_test = acc_test.item()
            best_dp_test = parity_test
            best_eo_test = equality_test

        print("Epoch {}:".format(epoch),
              "Loss tr: {:.4f}".format(loss.item()),
              "[Acc_sen0 tr: {:.4f}".format(acc_train.item()),
              "va: {:.4f}".format(acc_val.item()),
              "te: {:.4f}]".format(acc_test.item()),
              "[DP: {:.4f}".format(parity_test),
              "EO: {:.4f}]".format(equality_test),
              "{}/{:.4f}".format(best_epoch, best_test))

    print("Test results:",
          "[Acc: {:.4f}".format(best_test),
          "Auc: {:.4f}".format(best_auc_roc_test),
          "F1: {:.4f}]".format(best_f1_s_test),
          "[DP: {:.4f}".format(best_dp_test),
          "EO: {:.4f}]".format(best_eo_test))
    return best_test, best_auc_roc_test, best_f1_s_test, best_dp_test, best_eo_test


def main_worker(config):
    print(config)
    seed_everything(config['seed'])
    # device = 'cuda:{}'.format(config['cuda'])
    # torch.cuda.set_device(config['cuda'])

    E, U = e.detach().clone(), u.detach().clone()

    classifier0 = Classifier(config['signal_dim'],
                             config['signal_dim'],
                             1)
    net_sen0 = Specformer(1,
                          x.size(1),
                          config['nlayer'],
                          config['hidden_dim'],
                          config['signal_dim'],
                          config['num_heads'],
                          config['tran_dropout'],
                          config['feat_dropout'],
                          config['prop_dropout'],
                          config['norm'],
                          classifier=classifier0).cuda()
    classifier1 = Classifier(config['signal_dim'],
                             config['signal_dim'],
                             1)
    net_sen1 = Specformer(1,
                          x.size(1),
                          config['nlayer'],
                          config['hidden_dim'],
                          config['signal_dim'],
                          config['num_heads'],
                          config['tran_dropout'],
                          config['feat_dropout'],
                          config['prop_dropout'],
                          config['norm'],
                          classifier=classifier1).cuda()

    net_sen0.apply(init_params)
    net_sen1.apply(init_params)
    params = list(net_sen0.parameters()) + list(net_sen1.parameters())
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net_sen0))
    print(count_parameters(net_sen1))

    best_acc = 0.0
    best_loss = 1e5
    best_epoch = -1
    best_auc_roc_test = 0.0
    best_f1_s_test = 0.0
    best_test = 0.0
    best_dp_test = 1e5
    best_eo_test = 1e5
    for epoch in range(config['epoch_fit'] + config['epoch_debias']):
        net_sen0.train()
        net_sen1.train()
        optimizer.zero_grad()

        logit_sen0, H_sen0 = net_sen0(E, U, x)
        logit_sen1, H_sen1 = net_sen1(E, U, x)

        # cov = torch.tensor(0.0)
        group_distance = 0.0
        if epoch >= config['epoch_fit']:
            # group_distance = F.cosine_similarity(H_sen0, H_sen1).abs().mean().squeeze()
            # group_distance = linalg.vector_norm(H_sen0 - H_sen1, dim=1).mean() / (torch.std(H_sen0, dim=1).mean() + torch.std(H_sen1, dim=1).mean() + 1e-8)
            # group_distance = (linalg.vector_norm(H_sen0 - H_sen1, dim=1) / (torch.std(H_sen0, dim=1) + torch.std(H_sen1, dim=1) + 1e-8)).mean()
            # group_distance = linalg.vector_norm(H_sen0 - H_sen1) / (torch.std(H_sen0, dim=1).mean() + torch.std(H_sen1, dim=1).mean())

            if config['align_mode'] == 'l1':
                group_distance = (H_sen0 - H_sen1).abs().mean()  # pokec_n
            elif config['align_mode'] == 'l2':
                group_distance = linalg.vector_norm(H_sen0 - H_sen1, dim=1).mean()
            else:
                raise ValueError('Unknown group align!')

        loss_sen0 = F.binary_cross_entropy_with_logits(logit_sen0[idx_train_0], labels[idx_train_0].unsqueeze(1).float())
        loss_sen1 = F.binary_cross_entropy_with_logits(logit_sen1[idx_train_1], labels[idx_train_1].unsqueeze(1).float())
        loss = loss_sen1 + loss_sen0 + config['align_weight'] * group_distance
        # loss = loss_sen1 + loss_sen0 + config['cov'] * cov
        acc_train_sen0 = accuracy(logit_sen0[idx_train_0], labels[idx_train_0]) * 100.0
        acc_train_sen1 = accuracy(logit_sen1[idx_train_1], labels[idx_train_1]) * 100.0
        loss.backward()
        optimizer.step()

        net_sen0.eval()
        logit_sen0, _ = net_sen0(E, U, x)
        acc_val_sen0 = accuracy(logit_sen0[idx_val], labels[idx_val]) * 100.0
        acc_test_sen0 = accuracy(logit_sen0[idx_test], labels[idx_test]) * 100.0

        net_sen1.eval()
        logit_sen1, _ = net_sen1(E, U, x)
        acc_val_sen1 = accuracy(logit_sen1[idx_val], labels[idx_val]) * 100.0
        acc_test_sen1 = accuracy(logit_sen1[idx_test], labels[idx_test]) * 100.0

        parity_test, equality_test = fair_metric(logit_sen0, idx_test, labels, sens)
        auc_roc_test, f1_s_test, _ = evaluation_results(logit_sen0, labels, idx_test)
        parity_test, equality_test = parity_test * 100.0, equality_test * 100.0
        auc_roc_test, f1_s_test = auc_roc_test * 100.0, f1_s_test * 100.0

        # if loss_val < best_loss:
        #     best_loss = loss_val.item()
        if epoch > config['epoch_fit'] + config['patience'] and acc_val_sen0 > best_acc:
            best_acc = acc_val_sen0.item()
            best_epoch = epoch
            best_auc_roc_test = auc_roc_test.item()
            best_f1_s_test = f1_s_test.item()
            best_test = acc_test_sen0.item()
            best_dp_test = parity_test
            best_eo_test = equality_test

        print("Epoch {}:".format(epoch),
              "Loss tr: {:.4f}".format(loss.item()),
              "[Acc_sen0 tr: {:.4f}".format(acc_train_sen0.item()),
              "va: {:.4f}".format(acc_val_sen0.item()),
              "te: {:.4f}]".format(acc_test_sen0.item()),
              "[Acc_sen1 tr: {:.4f}".format(acc_train_sen1.item()),
              "va: {:.4f}".format(acc_val_sen1.item()),
              "te: {:.4f}]".format(acc_test_sen1.item()),
              "[DP: {:.4f}".format(parity_test),
              "EO: {:.4f}]".format(equality_test),
              "ali: {:.4f}".format(linalg.vector_norm(H_sen0 - H_sen1, dim=1).mean()),
              "std: {:.4f}".format(H_sen0.std(dim=1).mean()),
              "{}/{:.4f}".format(best_epoch, best_test))

    print("Test results:",
          "[Acc: {:.4f}".format(best_test),
          "Auc: {:.4f}".format(best_auc_roc_test),
          "F1: {:.4f}]".format(best_f1_s_test),
          "[DP: {:.4f}".format(best_dp_test),
          "EO: {:.4f}]".format(best_eo_test))

    # return fit_label(H_sen0)
    return best_test, best_auc_roc_test, best_f1_s_test, best_dp_test, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='bail')
    parser.add_argument('--rank', type=int, default=0, help="result stat")
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    config['seeds'] = args.seeds
    config['dataset'] = args.dataset
    config['rank'] = args.rank

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='../',
                                                                                   dataset=config['dataset'],
                                                                                   label_number=config['label_number'],
                                                                                   sens_number=config['sens_number'])
    assert (torch.equal(torch.abs(labels[idx_train] - 0.5) * 2.0, torch.ones_like(labels[idx_train])))
    assert (torch.equal(torch.abs(labels[idx_val] - 0.5) * 2.0, torch.ones_like(labels[idx_val])))
    assert (torch.equal(torch.abs(labels[idx_test] - 0.5) * 2.0, torch.ones_like(labels[idx_test])))
    assert (torch.equal(torch.abs(sens - 0.5) * 2.0, torch.ones_like(sens)))

    if torch.equal(idx_train, idx_sens_train) is not True:
        print("idx_train and idx_sens_train are not alignment, crop idx_train.")
        idx_train_crop = np.asarray(list(set(idx_train.cpu().numpy()) & set(idx_sens_train.cpu().numpy())))
        idx_train_crop = torch.LongTensor(idx_train_crop)
        idx_train_0, idx_train_1 = group_by_attr(idx_train_crop, sens)
    else:
        idx_train_0, idx_train_1 = group_by_attr(idx_train, sens)
    import random
    random.seed(20)
    random.shuffle(idx_train_0)
    random.shuffle(idx_train_1)
    if idx_train_0.shape[0] < idx_train_1.shape[0] or config['dataset'] == 'income':
        print("idx_train_0, idx_train_1 swap.")
        tmp = idx_train_0
        idx_train_0 = idx_train_1
        idx_train_1 = tmp
    print("idx_train: {}, idx_train_0: {}, idx_train_1: {}".format(idx_train.shape[0], idx_train_0.shape[0], idx_train_1.shape[0]))
    
    assert (len(list(set(idx_train_0.cpu().numpy()) & set(idx_train_1.cpu().numpy()))) == 0)
    assert (len(list(set(idx_train_0.cpu().numpy()) & set(idx_val.cpu().numpy()))) == 0)
    assert (len(list(set(idx_train_0.cpu().numpy()) & set(idx_test.cpu().numpy()))) == 0)
    assert (len(list(set(idx_train_1.cpu().numpy()) & set(idx_val.cpu().numpy()))) == 0)
    assert (len(list(set(idx_train_1.cpu().numpy()) & set(idx_test.cpu().numpy()))) == 0)

    e, u = [], []
    deg = np.array(adj.sum(axis=0)).flatten()
    for eps in config['eps']:
        print("Start building e, u with {}...".format(eps), end='')
        # build graph matrix
        D_ = sp.sparse.diags(deg ** eps)
        A_ = D_.dot(adj.dot(D_))
        # A_ = sp.sparse.eye(adj.shape[0]) - A_

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

    acc_test, best_auc_roc_test, best_f1_s_test, dp_test, eo_test = [], [], [], [], []
    for seed in config['seeds']:
        config['seed'] = seed
        _acc_test, _best_auc_roc_test, _best_f1_s_test, _dp_test, _eo_test = main_worker(config)
        acc_test.append(_acc_test)
        best_auc_roc_test.append(_best_auc_roc_test)
        best_f1_s_test.append(_best_f1_s_test)
        dp_test.append(_dp_test)
        eo_test.append(_eo_test)

    acc_test = np.array(acc_test, dtype=float)
    best_auc_roc_test = np.array(best_auc_roc_test, dtype=float)
    best_f1_s_test = np.array(best_f1_s_test, dtype=float)
    dp_test = np.array(dp_test, dtype=float)
    eo_test = np.array(eo_test, dtype=float)

    ACC = "{:.2f} $\pm$ {:.2f}".format(np.mean(acc_test), np.std(acc_test))
    AUC = "{:.2f} $\pm$ {:.2f}".format(np.mean(best_auc_roc_test), np.std(best_auc_roc_test))
    F1 = "{:.2f} $\pm$ {:.2f}".format(np.mean(best_f1_s_test), np.std(best_f1_s_test))
    DP = "{:.2f} $\pm$ {:.2f}".format(np.mean(dp_test), np.std(dp_test))
    EO = "{:.2f} $\pm$ {:.2f}".format(np.mean(eo_test), np.std(eo_test))
    print("Mean over {} run:".format(len(config['seeds'])),
          "Acc: " + ACC,
          "Auc: " + AUC,
          "F1: " + F1,
          "DP: " + DP,
          "EO: " + EO)

    result_append(ACC, AUC, F1, DP, EO, config)