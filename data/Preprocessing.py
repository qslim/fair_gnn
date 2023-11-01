import torch
from utils import load_credit, load_bail, load_german, feature_norm

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)


def load_data(path_root, dataset):
    # Load credit_scoring dataset
    if dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = path_root + "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    # Load german dataset
    elif dataset == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = path_root + "./dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )
    # Load bail dataset
    elif dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = path_root + "./dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr,
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    # elif dataset == 'synthetic':
    #     sens_idx = 0
    #     label_number = 1000
    #     path_sythetic = path_root + './dataset/synthetic.mat'
    #     adj, features, labels, idx_train, idx_val, idx_test, sens, raw_data_info = load_synthetic(path=path_sythetic,
    #                                                                           label_number=label_number)
    else:
        raise ValueError('Unknown dataset!')

    print("loaded dataset: ", dataset, "num of node: ", len(features), ' feature dim: ', features.shape[1])

    num_class = labels.unique().shape[0]-1
    return adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx

