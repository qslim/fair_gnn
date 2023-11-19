import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from specformer import Specformer
from data.Preprocessing import load_data
import scipy as sp
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric, evaluation_results, get_sens_idx, fair_metric_threshold_dp, fair_metric_threshold_eo, accuracy_threshold
torch.set_printoptions(profile='full')


def threshold_shfit(output, is_eo=False):
    # output = torch.sigmoid(output.squeeze())
    output = output.squeeze()
    sens_idx_0, sens_idx_1 = get_sens_idx(idx_sens_train, sens)
    if is_eo:
        _, sens_idx_0 = get_sens_idx(sens_idx_0, labels)
        _, sens_idx_1 = get_sens_idx(sens_idx_1, labels)
    sens_sorted_0, _ = output[sens_idx_0].sort()
    sens_sorted_1, _ = output[sens_idx_1].sort()
    print('Sens size: {}, {}'.format(sens_sorted_0.shape[0], sens_sorted_1.shape[0]))
    _threshold_sens0, _threshold_sens1 = -1, -1
    for i in range(sens_sorted_0.shape[0]):
        if sens_sorted_0[i] >= 0.0:
            _threshold_sens0 = i / sens_sorted_0.shape[0]
            print('Sens0 size: {}, {:.6f}'.format(i, _threshold_sens0))
            break
    for i in range(sens_sorted_1.shape[0]):
        if sens_sorted_1[i] >= 0.0:
            _threshold_sens1 = i / sens_sorted_1.shape[0]
            print('Sens1 size: {}, {:.6f}'.format(i, _threshold_sens1))
            break
    _threshold_sens = (_threshold_sens0 + _threshold_sens1) / 2.0
    print('Sens proportion: {:.6f}'.format(_threshold_sens))
    threshold_sens0, threshold_sens1 = sens_sorted_0[int(sens_sorted_0.shape[0] * _threshold_sens)].item(), sens_sorted_1[int(sens_sorted_1.shape[0] * _threshold_sens)].item()

    return threshold_sens0, threshold_sens1


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.cuda)

    E, U = e.detach().clone(), u.detach().clone()

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
    best_loss = 1e5
    best_epoch = -1
    best_auc_roc_test = 0.0
    best_f1_s_test = 0.0
    best_test = 0.0
    best_dp_test = 1e5
    best_eo_test = 1e5
    best_output = None
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output, _ = net(E, U, x)

        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        net.eval()
        output, _ = net(E, U, x)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        parity_test, equality_test = fair_metric(output, idx_test, labels, sens)
        auc_roc_test, f1_s_test, _ = evaluation_results(output, labels, idx_test)


        acc_val, acc_test, parity_test, equality_test = acc_val * 100.0, acc_test * 100.0, parity_test * 100.0, equality_test * 100.0
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
            best_output = output

        print("Epoch {}:".format(epoch),
              "loss: {:.4f}".format(loss.item()),
              "loss_v: {:.4f}".format(loss_val.item()),
              "acc_v: {:.4f}".format(acc_val.item()),
              "acc_t: {:.4f}".format(acc_test.item()),
              "auc_roc_t: {:.4f}".format(auc_roc_test.item()),
              "f1_s_t: {:.4f}".format(f1_s_test.item()),
              "[dp_t: {:.4f}".format(parity_test),
              "eo_t: {:.4f}]".format(equality_test),
              " {}/{:.4f}".format(best_epoch, best_test))

    dp_threshold_sens0, dp_threshold_sens1 = threshold_shfit(best_output, is_eo=False)
    print('dp_threshold_sens0: {}'.format(dp_threshold_sens0))
    print('dp_threshold_sens1: {}'.format(dp_threshold_sens1))
    parity_test_before = fair_metric_threshold_dp(best_output, idx_test, labels, sens, 0.0, 0.0)
    parity_test_after   = fair_metric_threshold_dp(best_output, idx_test, labels, sens, dp_threshold_sens0, dp_threshold_sens1)
    print('DP_test: {:.4f} -> {:.4f}'.format(parity_test_before * 100.0, parity_test_after * 100.0))

    # eo_threshold_sens0, eo_threshold_sens1 = threshold_shfit(best_output, is_eo=True)
    # print('eo_threshold_sens0: {}'.format(eo_threshold_sens0))
    # print('eo_threshold_sens1: {}'.format(eo_threshold_sens1))
    # equality_test_before = fair_metric_threshold_eo(best_output, idx_test, labels, sens, 0.0, 0.0)
    # equality_test_after   = fair_metric_threshold_eo(best_output, idx_test, labels, sens, eo_threshold_sens0, eo_threshold_sens1)
    # print('EO_test: {:.4f} -> {:.4f}'.format(equality_test_before * 100.0, equality_test_after * 100.0))

    acc_before = accuracy_threshold(best_output, idx_test, labels, sens, 0.0, 0.0).item()
    acc_after  = accuracy_threshold(best_output, idx_test, labels, sens, dp_threshold_sens0, dp_threshold_sens1).item()
    print('Acc_test: {:.4f} -> {:.4f}'.format(acc_before * 100.0, acc_after * 100.0))

    print("Test results:",
          "[acc_t= {:.4f}".format(acc_after),
          "auc_roc_t= {:.4f}".format(best_auc_roc_test),
          "f1_s_t= {:.4f}]".format(best_f1_s_test),
          "[dp_t: {:.4f}".format(parity_test_after),
          "eo_t: {:.4f}]".format(best_eo_test))
    return acc_after, best_auc_roc_test, best_f1_s_test, parity_test_after, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='income')
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='../',
                                                                                   dataset=args.dataset)
    assert (torch.equal(torch.abs(labels[idx_train] - 0.5) * 2.0, torch.ones_like(labels[idx_train])))
    assert (torch.equal(torch.abs(labels[idx_val] - 0.5) * 2.0, torch.ones_like(labels[idx_val])))
    assert (torch.equal(torch.abs(labels[idx_test] - 0.5) * 2.0, torch.ones_like(labels[idx_test])))
    assert (torch.equal(torch.abs(sens - 0.5) * 2.0, torch.ones_like(sens)))


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

    acc_test, best_auc_roc_test, best_f1_s_test, dp_test, eo_test = [], [], [], [], []
    for seed in args.seeds:
        args.seed = seed
        _acc_test, _best_auc_roc_test, _best_f1_s_test, _dp_test, _eo_test = main_worker(args, config)
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
    print("Mean over {} run:".format(len(args.seeds)),
          "[acc= {:.4f}_{:.4f}".format(np.mean(acc_test), np.std(acc_test)),
          "auc_roc= {:.4f}_{:.4f}".format(np.mean(best_auc_roc_test), np.std(best_auc_roc_test)),
          "f1_s= {:.4f}_{:.4f}]".format(np.mean(best_f1_s_test), np.std(best_f1_s_test)),
          "[dp: {:.4f}_{:.4f}".format(np.mean(dp_test), np.std(dp_test)),
          "eo: {:.4f}_{:.4f}]".format(np.mean(eo_test), np.std(eo_test))
          )

