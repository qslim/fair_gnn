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

    net_sens.apply(init_params)
    net.apply(init_params)
    params = list(net_sens.parameters()) + list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net_sens))
    print(count_parameters(net))

    best_acc = 0.0
    best_loss = 1e5
    best_epoch = -1
    best_auc_roc_test = 0.0
    best_f1_s_test = 0.0
    best_test = 0.0
    best_dp_test = 1e5
    best_eo_test = 1e5
    sens_acc_test, sens_acc_val = -1.0, -1.0
    for epoch in range(config['epoch_fit'] + config['epoch_debias']):
        net_sens.train()
        net.train()
        optimizer.zero_grad()

        output_sens, _ = net_sens(E, U, x)
        output, _ = net(E, U, x)

        # kl_div = F.mse_loss(torch.sigmoid(output), torch.sigmoid(output_sens))

        # output_g, output_sens_g = torch.cat((output, -output), dim=1), torch.cat((output_sens, -output_sens), dim=1)
        # kl_div = F.kl_div(F.logsigmoid(output_g), F.sigmoid(output_sens_g), reduction="batchmean")
        # # kl_div = F.kl_div(F.logsigmoid(output), F.sigmoid(output_sens), reduction="batchmean")

        # cov = torch.tensor(0.0)
        if epoch >= config['epoch_fit']:
            # y_score, s_score = torch.sigmoid(output), torch.sigmoid(output_sens)
            # cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

            # debias linearly
            output = output.squeeze()
            output_mean = output.mean()
            _output_sens = output_sens.squeeze()
            _output_sens = _output_sens - _output_sens.mean()
            output = ((output - output_mean) - config['orthogonality'] * ((output - output_mean) * _output_sens).sum() / (_output_sens.pow(2).sum() + 1e-8) * _output_sens + output_mean).unsqueeze(-1)

        loss_sens = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train],
                                                  sens[idx_sens_train].unsqueeze(1).float())
        loss_cls = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        loss = loss_cls + loss_sens
        # loss = loss_cls + loss_sens + config['cov'] * cov
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        net_sens.eval()
        output_sens, _ = net_sens(E, U, x)
        acc_val_sens = accuracy(output_sens[idx_val], sens[idx_val])
        acc_test_sens = accuracy(output_sens[idx_test], sens[idx_test])
        acc_val_sens, acc_test_sens = acc_val_sens * 100.0, acc_test_sens * 100.0

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
        if epoch > config['epoch_fit'] + 5 and acc_val > best_acc:
            best_acc = acc_val.item()
            best_epoch = epoch
            best_auc_roc_test = auc_roc_test.item()
            best_f1_s_test = f1_s_test.item()
            best_test = acc_test.item()
            best_dp_test = parity_test
            best_eo_test = equality_test

        print("Epoch {}:".format(epoch),
              # "cov: {:.4f}".format(cov.item()),
              "loss: {:.4f}".format(loss.item()),
              "loss_v: {:.4f}".format(loss_val.item()),
              "acc_v: {:.4f}".format(acc_val.item()),
              "acc_t: {:.4f}".format(acc_test.item()),
            #   "auc_roc_t: {:.4f}".format(auc_roc_test.item()),
            #   "f1_s_t: {:.4f}".format(f1_s_test.item()),
              "[dp_t: {:.4f}".format(parity_test),
              "eo_t: {:.4f}]".format(equality_test),
              " {}/{:.4f}".format(best_epoch, best_test),
              "sens_acc_v: {:.4f}".format(acc_val_sens),
              "sens_acc_t= {:.4f}]".format(acc_test_sens))

    print("Test results:",
          "sens_acc_v: {:.4f}".format(sens_acc_val),
          "sens_acc_t= {:.4f}]".format(sens_acc_test),
          "[acc_t= {:.4f}".format(best_test),
          "auc_roc_t= {:.4f}".format(best_auc_roc_test),
          "f1_s_t= {:.4f}]".format(best_f1_s_test),
          "[dp_t: {:.4f}".format(best_dp_test),
          "eo_t: {:.4f}]".format(best_eo_test))
    return best_test, best_auc_roc_test, best_f1_s_test, best_dp_test, best_eo_test, sens_acc_val, sens_acc_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
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

    acc_test, best_auc_roc_test, best_f1_s_test, dp_test, eo_test, sens_acc_val, sens_acc_test = [], [], [], [], [], [], []
    for seed in args.seeds:
        args.seed = seed
        _acc_test, _best_auc_roc_test, _best_f1_s_test, _dp_test, _eo_test, _sens_acc_val, _sens_acc_test = main_worker(args, config)
        acc_test.append(_acc_test)
        best_auc_roc_test.append(_best_auc_roc_test)
        best_f1_s_test.append(_best_f1_s_test)
        dp_test.append(_dp_test)
        eo_test.append(_eo_test)
        sens_acc_val.append(_sens_acc_val)
        sens_acc_test.append(_sens_acc_test)

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
          "eo: {:.4f}_{:.4f}]".format(np.mean(eo_test), np.std(eo_test)),
          "[s_acc_v: {:.4f}_{:.4f}".format(np.mean(sens_acc_val), np.std(sens_acc_val)),
          "s_acc_t: {:.4f}_{:.4f}]".format(np.mean(sens_acc_test), np.std(sens_acc_test))
          )

