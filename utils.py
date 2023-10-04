import time
import math
import random
import numpy as np
import scipy as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import networkx as nx
import sklearn.preprocessing as skpp
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import pandas as pd
import scipy.sparse as spp


# from scipy.sparse import csr_matrix


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(dataset, y, nclass, seed=0):
    if dataset == 'arxiv':
        dataset = DglNodePropPredDataset('ogbn-arxiv')
        split = dataset.get_idx_split()
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    elif dataset == 'penn':
        split = np.load('node_raw_data/fb100-Penn94-splits.npy', allow_pickle=True)[0]
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    else:
        y = y.cpu()

        percls_trn = int(round(0.6 * len(y) / nclass))
        val_lb = int(round(0.2 * len(y)))

        indices = []
        for i in range(nclass):
            index = (y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0), device=index.device)]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_index = rest_index[:val_lb]
        test_index = rest_index[val_lb:]

        return train_index, valid_index, test_index


def calculate_similarity_matrix(adj, features, metric=None, filterSigma=None, normalize=None, largestComponent=False):
    if metric in ['cosine', 'jaccard']:
        # build similarity matrix
        if largestComponent:
            graph = nx.from_scipy_sparse_matrix(adj)
            lcc = max(nx.connected_components(graph), key=len)  # take largest connected components
            adj = nx.to_scipy_sparse_matrix(graph, nodelist=lcc, dtype='float', format='csc')
        sim = get_similarity_matrix(adj, metric=metric)
        if filterSigma:
            sim = filter_similarity_matrix(sim, sigma=filterSigma)
        if normalize:
            sim = symmetric_normalize(sim)
    return sim


def get_similarity_matrix(mat, metric=None):
    """
    get similarity matrix of nodes in specified metric
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :param metric: similarity metric
    :return: similarity matrix of nodes
    """
    if metric == 'jaccard':
        return jaccard_similarity(mat.tocsc())
    elif metric == 'cosine':
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError('Please specify the type of similarity metric.')


def filter_similarity_matrix(sim, sigma):
    """
    filter value by threshold = mean(sim) + sigma * std(sim)
    :param sim: similarity matrix
    :param sigma: hyperparameter for filtering values
    :return: filtered similarity matrix
    """
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = sp.sparse.diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees


def jaccard_similarity(mat):
    """
    get jaccard similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= (aa + bb - ab.data)
    return sim


def cosine_similarity(mat):
    """
    get cosine similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    mat_row_norm = skpp.normalize(mat, axis=1)  # 对mat进行正则化
    sim = mat_row_norm.dot(mat_row_norm.T)  # .dot点积
    return sim


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/",
                label_number=1000):  # 写了=的，就是不传输的时候的默认值，如果没有=的，则必须要传对应值。上传的时候可以用=表示自己要传的是哪个特定参数
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))  # "{}.csv".format(“XXX”)生成着"XXX.csv"，os.path.join(A,B)拼接A和B在一起
    header = list(idx_features_labels.columns)  # list将括号里的内容变为数组
    header.remove(predict_attr)  # header.remove删除括号内的东西
    header.remove('user_id')
    '''
    #    # Normalize MaxBillAmountOverLast6Months
    #    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
    #
    #    # Normalize MaxPaymentAmountOverLast6Months
    #    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
    #
    #    # Normalize MostRecentBillAmount
    #    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
    #
    #    # Normalize MostRecentPaymentAmount
    #    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
    #
    #    # Normalize TotalMonthsOverdue
    #    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()
    '''
    # build relationship
    # if os.path.exists(f'{path}/{dataset}_edges.txt'):  # os.path.exists判断文件是否存在,f让大括号部分，用里面的内容取代
    #     edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype(
    #         'int')  # np.genfromtxt读txt的方法，astype表示读成数字
    # else:
    #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)  # 用build的方式生成的数组存储进edge
    #     np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)  # 将数据存进文件

    features = spp.csr_matrix(idx_features_labels[header],
                              dtype=np.float32)  # sp.csr_matrix压缩稀疏矩阵（https://www.runoob.com/scipy/scipy-sparse-matrix.html）
    labels = idx_features_labels[predict_attr].values  # 存下predict_attr的数值
    idx = np.arange(features.shape[0])  # 0到features：{0，1，2，...，features.shape[0]-1}
    idx_map = {j: i for i, j in enumerate(idx)}  # {0:0, 1:1, 2:2, ... , feature.shape[0]-1:feature.shape[0]-1}
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  # flatten()拉成一维数组，map把第二个的变量添加到第一个里面去，list(map())将edge_unordered.flatten()的值作为key，依次返回对应的value
    #                  dtype=int).reshape(edges_unordered.shape)  # 将数据拆分成edges_unordered大小的行数的矩阵
    # adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                  shape=(labels.shape[0], labels.shape[0]),
    #                  dtype=np.float32)  # 视sp.coo_matrix生成稀疏矩阵（与csr_matrix相反）
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 相似矩阵
    # adj = adj + spp.eye(adj.shape[0])  # sp.eye对角线上位1的矩阵

    features = torch.FloatTensor(np.array(features.todense()))  # 将numpy转换为tensor（32位浮点类型数据）
    labels = torch.LongTensor(labels)  # 将数据转化为64位整形

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]  # labels里满足0的索引
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)  # 洗牌
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])  # 指定训练集
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])  # 标定验证集
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])  # 指定测试集

    # print(f"Unique labels: {torch.unique(labels)}") #打印labels里不重复的参数
    # print(f"Label size by class: {len(label_idx_0), len(label_idx_1)}") #打印了0的个数和1的个数

    # print(f"The size of the training set is: {len(idx_train)} and max size of training set is { int(0.5 * len(label_idx_0)) + int(0.5 * len(label_idx_1)) }") #打印训练集的数量；打印理论最大训练集数量
    # print(f"The size of the validation set is: {len(idx_val)}") #打印验证集的数量
    # print(f"The size of the test set is: {len(idx_test)}") #打印测试集的数量

    sens = idx_features_labels[sens_attr].values.astype(int)  # 把sensitive attribute记录成int
    sens = torch.FloatTensor(sens)  # 转换数据类型
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    torch.save([features, labels, idx_train, idx_val, idx_test, sens], 'data/{}_information.pt'.format(dataset))
    return None, features, labels, idx_train, idx_val, idx_test, sens


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns,
                             index=x.T.columns)  # distance_matrix 计算任意两行之间的距离，第i行到第j行的距离返回在ij
    df_euclid = df_euclid.to_numpy()  # 格式转换
    idx_map = []  # 用来记录的list
    for ind in range(df_euclid.shape[0]):  # numpy.shape是数组的维度，numpy.shape[0]是矩阵的行数
        max_sim = np.sort(df_euclid[ind, :])[-2]  # 每一行取第二大的
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]  # 满足比tresh*max_sim大的数
        import random
        random.seed(912)
        random.shuffle(neig_id)  # 把neig打乱
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])  # 把neig不等于行数的存在idx_map中
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def convert_sparse_matrix_to_sparse_tensor(X):
    X = X.tocoo()  # 将稠密矩阵转化为稀疏矩阵

    X = torch.sparse_coo_tensor(torch.tensor([X.row.tolist(), X.col.tolist()]),
                                torch.tensor(X.data.astype(np.float32)))  # 创建稀疏tensor
    return X