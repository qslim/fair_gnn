import random
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn.functional as F


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


def evaluation_results(output, labels, idx):
    preds = (output.squeeze() > 0).type_as(labels)
    auc_roc = roc_auc_score(labels.cpu().numpy()[idx.cpu().numpy()], output.detach().cpu().numpy()[idx.cpu().numpy()])
    f1_s = f1_score(labels[idx.cpu().numpy()].cpu().numpy(), preds[idx.cpu().numpy()].cpu().numpy())
    acc = accuracy_score(labels[idx.cpu().numpy()].cpu().numpy(), preds[idx.cpu().numpy()].cpu().numpy())
    return auc_roc, f1_s, acc


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


def group_by_attr(idx, attr):
    sens_idx_0 = np.where(attr.cpu() == 0)[0]
    sens_idx_1 = np.where(attr.cpu() == 1)[0]
    sens_idx_0 = np.asarray(list(set(idx.cpu().numpy()) & set(sens_idx_0)))
    sens_idx_1 = np.asarray(list(set(idx.cpu().numpy()) & set(sens_idx_1)))
    sens_idx_0 = torch.LongTensor(sens_idx_0)
    sens_idx_1 = torch.LongTensor(sens_idx_1)
    return sens_idx_0, sens_idx_1


def fair_metric_threshold_dp(output, idx, labels, sens, threshold0, threshold1):
    sens_idx_0, sens_idx_1 = group_by_attr(idx, sens)

    pred_y_0 = (output[sens_idx_0].squeeze() > threshold0).type_as(labels).cpu().numpy()
    pred_y_1 = (output[sens_idx_1].squeeze() > threshold1).type_as(labels).cpu().numpy()

    parity = abs(sum(pred_y_0) / sens_idx_0.shape[0] - sum(pred_y_1) / sens_idx_1.shape[0])

    return parity


def fair_metric_threshold_eo(output, idx, labels, sens, threshold0, threshold1):
    sens_idx_0, sens_idx_1 = group_by_attr(idx, sens)
    _, sens0_label1_idx = group_by_attr(sens_idx_0, labels)
    _, sens1_label1_idx = group_by_attr(sens_idx_1, labels)

    sens0_label1_pred = (output[sens0_label1_idx].squeeze() > threshold0).type_as(labels).cpu().numpy()
    sens1_label1_pred = (output[sens1_label1_idx].squeeze() > threshold1).type_as(labels).cpu().numpy()

    equality = abs(sum(sens0_label1_pred) / sens0_label1_idx.shape[0] - sum(sens1_label1_pred) / sens1_label1_idx.shape[0])

    return equality


def accuracy_threshold(output, idx, labels, sens, threshold0, threshold1):
    output = output.squeeze()

    sens_idx_0, sens_idx_1 = group_by_attr(idx, sens)

    preds_0 = (output[sens_idx_0] > threshold0).type_as(labels)
    correct_0 = preds_0.eq(labels[sens_idx_0]).double()

    preds_1 = (output[sens_idx_1] > threshold1).type_as(labels)
    correct_1 = preds_1.eq(labels[sens_idx_1]).double()

    correct = correct_0.sum() + correct_1.sum()

    return correct / (sens_idx_0.shape[0] + sens_idx_1.shape[0])


def cosine_similarity(output, sens, idx):
    _output, _sens = output.squeeze()[idx], sens.squeeze()[idx].float()
    _output, _sens = _output - torch.mean(_output), _sens - torch.mean(_sens)
    return F.cosine_similarity(_output.unsqueeze(0), _sens.unsqueeze(0)).squeeze().abs()
