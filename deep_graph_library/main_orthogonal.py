import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import sys
sys.path.append('..')
from deep_graph_library.models import GCN, GAT
from data.Preprocessing import load_data
import scipy as sp
from utils import seed_everything, init_params, count_parameters, accuracy, fair_metric, evaluation_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_sens(g, x):
    net_sens = GCN(nfeat=x.size(1), nhid=config['hidden_dim'], nclass=1, dropout=config['feat_dropout']).cuda()
    # net_sens = GAT(num_layers=2, in_dim=x.size(1), num_hidden=config['hidden_dim'], num_classes=1, heads=1, feat_drop=config['feat_dropout'], attn_drop=config['feat_dropout'], negative_slope=0.2, residual=False).cuda()
    # net_sens = SGConv(in_feats=x.size(1), out_feats=1, k=2, cached=True, bias=True).cuda()

    net_sens.apply(init_params)
    optimizer = torch.optim.Adam(net_sens.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    print(count_parameters(net_sens))

    best_acc = 0.0
    best_epoch = -1
    best_test = 0.0
    for epoch in range(config['epoch']):
        net_sens.train()
        optimizer.zero_grad()
        output_sens, _ = net_sens(g, x)
        loss = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
        acc_train = accuracy(output_sens[idx_sens_train], sens[idx_sens_train])
        loss.backward()
        optimizer.step()

        net_sens.eval()
        output_sens, _ = net_sens(g, x)
        acc_val = accuracy(output_sens[idx_val], sens[idx_val])
        acc_test = accuracy(output_sens[idx_test], sens[idx_test])

        acc_val, acc_test = acc_val * 100.0, acc_test * 100.0

        if acc_val > best_acc:
            best_epoch = epoch
            best_acc = acc_val.item()
            best_test = acc_test.item()

    return output_sens, best_test, best_acc, best_epoch


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.cuda)

    if config['orthogonality'] != 0.0:
        output_sens, sens_acc_test, sens_acc_val, sens_epoch = fit_sens(g, x)

        output_sens = output_sens.detach()
        output_sens = output_sens.squeeze()

        # output_sens = torch.sigmoid(output_sens)
        # output_sens[idx_sens_train] = sens[idx_sens_train]

        _sens_gt = torch.max(torch.abs(output_sens))
        _sens = torch.where(sens == 1.0, _sens_gt, -_sens_gt)
        output_sens[idx_sens_train] = _sens[idx_sens_train]

        output_sens = output_sens - output_sens.mean()
    else:
        output_sens, sens_acc_test, sens_acc_val, sens_epoch = None, -1.0, -1.0, -1

    net = GCN(nfeat=x.size(1), nhid=config['hidden_dim'], nclass=1, dropout=config['feat_dropout']).cuda()
    # net = GAT(num_layers=2, in_dim=x.size(1), num_hidden=config['hidden_dim'], num_classes=1, heads=1, feat_drop=config['feat_dropout'], attn_drop=config['feat_dropout'], negative_slope=0.2, residual=False).cuda()
    # net = SGConv(in_feats=x.size(1), out_feats=1, k=2, cached=True, bias=True).cuda()
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
    for epoch in range(config['epoch']):
        net.train()
        optimizer.zero_grad()
        output, _ = net(g, x)

        if config['orthogonality'] != 0.0:
            # debias linearly
            output = output.squeeze()
            output_mean = output.mean()
            output = ((output - output_mean) - config['orthogonality'] * ((output - output_mean) * output_sens).sum() / (output_sens.pow(2).sum() + 1e-8) * output_sens + output_mean).unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        net.eval()
        output, _ = net(g, x)
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

    print("Test results:",
          "[sens_epoch: {}".format(sens_epoch),
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
    parser.add_argument('--seeds', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='pokec_n')
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='../',
                                                                                   dataset=args.dataset)
    assert (torch.equal(torch.abs(labels[idx_train] - 0.5) * 2.0, torch.ones_like(labels[idx_train])))
    assert (torch.equal(torch.abs(labels[idx_val] - 0.5) * 2.0, torch.ones_like(labels[idx_val])))
    assert (torch.equal(torch.abs(labels[idx_test] - 0.5) * 2.0, torch.ones_like(labels[idx_test])))
    assert (torch.equal(torch.abs(sens - 0.5) * 2.0, torch.ones_like(sens)))

    g = dgl.from_scipy(adj)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(torch.device(device))

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

