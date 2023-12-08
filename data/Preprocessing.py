import torch
from data.utils import load_credit, load_bail, load_german, load_income, load_pokec, feature_norm

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)


def load_data(path_root, dataset, label_number, sens_number):
    # Load credit_scoring dataset
    if dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        # label_number = 6000
        path_credit = path_root + "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        idx_sens_train = idx_train

    # Load german dataset
    elif dataset == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        # label_number = 100
        path_german = path_root + "./dataset/german"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )
        idx_sens_train = idx_train
    # Load bail dataset
    elif dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        # label_number = 100
        path_bail = path_root + "./dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr,
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )
        norm_features = feature_norm(features)
        # norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        idx_sens_train = idx_train

    elif dataset == 'income':
        sens_attr = "race"  # column number after feature process is 1
        sens_idx = 8
        predict_attr = 'income'
        # label_number = 3000
        path_income = path_root + "./dataset/income"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_income(dataset, sens_attr,
                                                                                     predict_attr, path=path_income,
                                                                                     label_number=label_number)
        norm_features = feature_norm(features)
        # norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
        idx_sens_train = idx_train
    # elif dataset == 'synthetic':
    #     sens_idx = 0
    #     label_number = 1000
    #     path_sythetic = path_root + './dataset/synthetic.mat'
    #     adj, features, labels, idx_train, idx_val, idx_test, sens, raw_data_info = load_synthetic(path=path_sythetic,
    #                                                                           label_number=label_number)
    elif dataset in ['nba', 'pokec_z', 'pokec_n']:
        # Load the dataset and split
        if dataset != 'nba':
            if dataset == 'pokec_z':
                dataset = 'region_job'
            else:
                dataset = 'region_job_2'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            # label_number = 500
            # sens_number = 200
            seed = 20
            path = "../dataset/pokec/"
        else:
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            # label_number = 100
            # sens_number = 50
            seed = 20
            path = "../dataset/NBA"

        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                        sens_attr,
                                                                                        predict_attr,
                                                                                        path=path,
                                                                                        label_number=label_number,
                                                                                        sens_number=sens_number,
                                                                                        seed=seed)
        # x = feature_norm(x)
        labels[labels > 1] = 1
        sens[sens > 0] = 1

    else:
        raise ValueError('Unknown dataset!')

    print("loaded dataset:", dataset, "num_node:", len(features), 'feature_dim:', features.shape[1], 'num_label:', idx_train.shape[0], 'num_sens:', idx_sens_train.shape[0])

    # num_class = labels.unique().shape[0]-1

    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

