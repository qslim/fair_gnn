import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from specformer_dual import Specformer_wrapper
from data.Preprocessing import load_data
from data.fairgraph_dataset2 import POKEC, NBA
import scipy as sp
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric, evaluation_results
from result_stat.result_append import result_append


def signal_debias(output, output_sens):
    # y_score, s_score = torch.sigmoid(output), torch.sigmoid(output_sens)
    # cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

    output = output.squeeze()
    output_mean = output.mean()
    _output_sens = output_sens.squeeze()
    _output_sens = _output_sens - _output_sens.mean()
    output = ((output - output_mean) - config['orthogonality'] * ((output - output_mean) * _output_sens).sum() / (
                _output_sens.pow(2).sum() + 1e-8) * _output_sens + output_mean).unsqueeze(-1)
    return output


def main_worker(config):
    print(config)
    seed_everything(config['seed'])
    # device = 'cuda:{}'.format(config['cuda'])
    # torch.cuda.set_device(config['cuda'])

    E, U = e.detach().clone(), u.detach().clone()

    net = Specformer_wrapper(nclass=1,
                             nfeat=x.size(1),
                             nlayer=config['nlayer'],
                             hidden_dim=config['hidden_dim'],
                             signal_dim=config['decorrela_dim'],
                             nheads=config['num_heads'],
                             tran_dropout=config['tran_dropout'],
                             feat_dropout=config['feat_dropout'],
                             prop_dropout=config['prop_dropout'],
                             shd_filter=config['shd_filter'] == 'T',
                             shd_trans=config['shd_trans'] == 'T').cuda()
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
    for epoch in range(config['epoch_fit'] + config['epoch_debias']):
        net.train()
        optimizer.zero_grad()

        output, output_sens = net(E, U, x)

        # cov = torch.tensor(0.0)
        if epoch >= config['epoch_fit']:
            output = signal_debias(output, output_sens)

        loss_sens = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train],
                                                  sens[idx_sens_train].unsqueeze(1).float())
        loss_cls = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        loss = loss_cls + loss_sens
        # loss = loss_cls + loss_sens + config['cov'] * cov
        acc_train_sens = accuracy(output_sens[idx_train], sens[idx_train]) * 100.0
        acc_train = accuracy(output[idx_train], labels[idx_train]) * 100.0
        loss.backward()
        optimizer.step()

        net.eval()
        output, output_sens = net(E, U, x)
        acc_val_sens = accuracy(output_sens[idx_val], sens[idx_val]) * 100.0
        acc_test_sens = accuracy(output_sens[idx_test], sens[idx_test]) * 100.0

        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        acc_val = accuracy(output[idx_val], labels[idx_val]) * 100.0
        acc_test = accuracy(output[idx_test], labels[idx_test]) * 100.0
        parity_test, equality_test = fair_metric(output, idx_test, labels, sens)
        auc_roc_test, f1_s_test, _ = evaluation_results(output, labels, idx_test)

        parity_test, equality_test = parity_test * 100.0, equality_test * 100.0
        auc_roc_test, f1_s_test = auc_roc_test * 100.0, f1_s_test * 100.0

        # if loss_val < best_loss:
        #     best_loss = loss_val.item()
        if epoch > config['epoch_fit'] + config['patience'] and acc_val > best_acc:
            best_acc = acc_val.item()
            best_epoch = epoch
            best_auc_roc_test = auc_roc_test.item()
            best_f1_s_test = f1_s_test.item()
            best_test = acc_test.item()
            best_dp_test = parity_test
            best_eo_test = equality_test

        print("Epoch {}:".format(epoch),
              # "cov: {:.4f}".format(cov.item()),
              "[Loss tr: {:.4f}".format(loss.item()),
              "va: {:.4f}]".format(loss_val.item()),
              "[Acc tr: {:.4f}".format(acc_train.item()),
              "va: {:.4f}".format(acc_val.item()),
              "te: {:.4f}]".format(acc_test.item()),
              # "auc_roc_t: {:.4f}".format(auc_roc_test.item()),
              # "f1_s_t: {:.4f}".format(f1_s_test.item()),
              "[DP: {:.4f}".format(parity_test),
              "EO: {:.4f}]".format(equality_test),
              "[Sen-acc tr: {:.4f}".format(acc_train_sens),
              "va: {:.4f}]".format(acc_val_sens),
              "te: {:.4f}]".format(acc_test_sens),
              "{}/{:.4f}".format(best_epoch, best_test))

    print("Test results:",
          "[Acc: {:.4f}".format(best_test),
          "Auc: {:.4f}".format(best_auc_roc_test),
          "F1: {:.4f}]".format(best_f1_s_test),
          "[DP: {:.4f}".format(best_dp_test),
          "EO: {:.4f}]".format(best_eo_test))
    return best_test, best_auc_roc_test, best_f1_s_test, best_dp_test, best_eo_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='pokec_z')
    parser.add_argument('--rank', type=int, default=0, help="result stat")
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    config['seeds'] = args.seeds
    config['dataset'] = args.dataset
    config['rank'] = args.rank

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='../',
                                                                                   dataset=config['dataset'])
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