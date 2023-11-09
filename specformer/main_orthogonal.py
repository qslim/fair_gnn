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
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric, evaluation_results


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.cuda)

    E, U = e.detach().clone(), u.detach().clone()

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
    best_epoch = -1
    for epoch in range(config['epoch']):
        net_sens.train()
        optimizer.zero_grad()
        output_sens, _ = net_sens(E, U, x)
        loss = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
        acc_train = accuracy(output_sens[idx_sens_train], sens[idx_sens_train])
        loss.backward()
        optimizer.step()

        net_sens.eval()
        output_sens, _ = net_sens(E, U, x)
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

    output_sens = output_sens.detach()
    output_sens = output_sens.squeeze()

    # output_sens = torch.sigmoid(output_sens)
    # output_sens[idx_sens_train] = sens[idx_sens_train]

    _sens_gt = torch.max(torch.abs(output_sens))
    assert (torch.equal(torch.abs(sens - 0.5) * 2.0, torch.ones_like(sens)))
    _sens = torch.where(sens == 1.0, _sens_gt, -_sens_gt)
    output_sens[idx_sens_train] = _sens[idx_sens_train]

    output_sens = output_sens - output_sens.mean()

    best_acc = 0.0
    best_loss = 1e5
    best_epoch = -1
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output, _ = net(E, U, x)

        # debias linearly
        output = output.squeeze()
        # output = (output - 1.0 * (output * output_sens).sum() / (output_sens.norm(dim=0) + 1e-8) * output_sens).unsqueeze(-1)
        output_mean = output.mean()
        output = ((output - output_mean) - 0.5 * 383.8109 * ((output - output_mean) * output_sens).sum() / (output_sens.pow(2).sum() + 1e-8) * output_sens + output_mean).unsqueeze(-1)

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

    print("Test results:",
          "acc_v: {:.4f}".format(best_acc),
          "acc_t= {:.4f}".format(best_test),
          "auc_roc_t= {:.4f}".format(best_auc_roc_test),
          "f1_s_t= {:.4f}".format(best_f1_s_test),
          "[dp_t: {:.4f}".format(best_dp_test),
          "eo_t: {:.4f}]".format(best_eo_test))
    return best_test, best_auc_roc_test, best_f1_s_test, best_dp_test, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='credit')
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='../',
                                                                                   dataset=args.dataset)


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
          "acc= {:.4f}_{:.4f}".format(np.mean(acc_test), np.std(acc_test)),
          "auc_roc= {:.4f}_{:.4f}".format(np.mean(best_auc_roc_test), np.std(best_auc_roc_test)),
          "f1_s= {:.4f}_{:.4f}".format(np.mean(best_f1_s_test), np.std(best_f1_s_test)),
          "[dp: {:.4f}_{:.4f}".format(np.mean(dp_test), np.std(dp_test)),
          "eo: {:.4f}_{:.4f}]".format(np.mean(eo_test), np.std(eo_test)))

