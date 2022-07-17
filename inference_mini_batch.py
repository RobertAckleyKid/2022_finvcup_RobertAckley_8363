# dataset name: XYGraphP1_no_valid

import pickle
from utils import XYGraphP1_no_valid
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np


from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib


sage_neighsampler_parameters = {'lr': 0.003, 'num_layers': 2, 'hidden_channels': 128, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7
                                }

gat_neighsampler_parameters = {'lr': 0.003, 'num_layers': 2, 'hidden_channels': 128, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-7, 'layer_heads': [4, 1]
                               }

gatv2_neighsampler_parameters = {'lr': 0.003, 'num_layers': 2, 'hidden_channels': 128, 'dropout': 0.0, 'batchnorm': False, 'l2': 5e-6, 'layer_heads': [4, 1]
                                 }


@torch.no_grad()
def to_embedding(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    out = model.to_embedding(data.x, layer_loader, device)
    print("Model embedding data : ", out.shape)

    return out


def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='XYGraphP1')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    print(args)

    no_conv = False
    if args.model in ['mlp']:
        no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = XYGraphP1_no_valid(
        root='./', name='xydata', transform=T.ToSparseTensor())

    nlabels = dataset.num_classes
    if args.dataset == 'XYGraphP1':
        nlabels = 2

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    if args.dataset in ['XYGraphP1']:
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    data = data.to(device)

    layer_loader = NeighborSampler(
        data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)

    if args.model == 'sage_neighsampler':
        para_dict = sage_neighsampler_parameters
        model_para = sage_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE_NeighSampler(
            in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    if args.model == 'gat_neighsampler':
        para_dict = gat_neighsampler_parameters
        model_para = gat_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GAT_NeighSampler(
            in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    if args.model == 'gatv2_neighsampler':
        para_dict = gatv2_neighsampler_parameters
        model_para = gatv2_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GATv2_NeighSampler(
            in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    model_file = './model_files/{}/{}/model.pt'.format(
        args.dataset, args.model)
    print('model_file:', model_file)
    model.load_state_dict(torch.load(model_file))

    out = to_embedding(layer_loader, model, data, device, no_conv)

    embedding_train, embedding_test = out[data.train_mask], out[data.test_mask]
    y_train = data.y[data.train_mask]

    print(embedding_train.shape)
    print(y_train.shape)
    print(embedding_test.shape)

    with open("xydata/embedding/64_train_embedding_new.pickle", 'wb') as file:
        pickle.dump(embedding_train, file)
    file.close()

    with open("xydata/embedding/64_embedding_test_new.pickle", 'wb') as file:
        pickle.dump(embedding_test, file)
    file.close()

    with open("xydata/embedding/64_y_train_new.pickle", 'wb') as file:
        pickle.dump(y_train, file)
    file.close()

    with open('xydata/embedding/64_embedding_test_new.pickle', 'rb') as file:
        x_test_all = pickle.load(file)
        file.close()
    with open('xydata/embedding/64_train_embedding_new.pickle', 'rb') as file:
        x_train_all = pickle.load(file)
        file.close()
    with open('xydata/embedding/64_y_train_new.pickle', 'rb') as file:
        y_train_all = pickle.load(file)
        file.close()

    items = np.load('xydata/raw/phase1_gdata.npz')
    x = items['x']
    y = items['y'].reshape(-1, 1)
    edge_index = items['edge_index']
    edge_type = items['edge_type']
    np.random.seed(42)
    train_mask_t = items['train_mask']
    np.random.shuffle(train_mask_t)
    test_mask = items['test_mask']

    x_train_add = torch.tensor(x[train_mask_t], dtype=torch.float).contiguous()
    x_test_add = torch.tensor(x[test_mask], dtype=torch.float).contiguous()

    set_train_mask_t = set(list(train_mask_t))
    set_test_mask_t = set(list(test_mask))
    set_used = set_train_mask_t | set_test_mask_t

    x_edge_add = np.zeros([x.shape[0], 22])

    edge_type = edge_type-1

    x_time_add = np.zeros([x.shape[0], 90])
    edge_timestamp = items['edge_timestamp']
    edge_timestamp = (edge_timestamp.astype(np.int32) / 13).astype(np.int32)

    x_pointLabel_add = np.zeros([x.shape[0], 8])

    for i in tqdm(range(len(edge_index))):
        if edge_index[i][0] in set_used or edge_index[i][1] in set_used:
            x_edge_add[edge_index[i][0]][edge_type[i]] += 1
            x_edge_add[edge_index[i][1]][edge_type[i]+10] += 1

            x_time_add[edge_index[i][0]][:edge_timestamp[i]+1] += 1
            x_time_add[edge_index[i][1]][45:edge_timestamp[i]+1+45] += 1

            if y[edge_index[i][1]] != -100:
                x_pointLabel_add[edge_index[i][0]][y[edge_index[i][1]]] += 1
            else:
                x_pointLabel_add[edge_index[i][0]][:2] += 1

            if y[edge_index[i][0]] != -100:
                x_pointLabel_add[edge_index[i][1]][y[edge_index[i][0]]+4] += 1
            else:
                x_pointLabel_add[edge_index[i][1]][4:6] += 1

    train_x_edge_add = torch.from_numpy(x_edge_add[train_mask_t])
    test_x_edge_add = torch.from_numpy(x_edge_add[test_mask])

    train_x_time_add = torch.from_numpy(x_time_add[train_mask_t])
    test_x_time_add = torch.from_numpy(x_time_add[test_mask])

    train_x_pointLabel_add = torch.from_numpy(x_pointLabel_add[train_mask_t])
    test_x_pointLabel_add = torch.from_numpy(x_pointLabel_add[test_mask])

    x_train_all = torch.cat((x_train_add, x_train_all, train_x_edge_add,
                            train_x_time_add, train_x_pointLabel_add), 1)
    x_test_all = torch.cat(
        (x_test_add, x_test_all, test_x_edge_add, test_x_time_add, test_x_pointLabel_add), 1)

    X_train, y_train = x_train_all, y_train_all

    gbm = LGBMClassifier(objective='binary',
                         # subsample=0.8,
                         # colsample_bytree=0.8,
                         verbosity=2, metric='auc',
                         learning_rate=0.01,
                         n_estimators=1200,
                         min_child_samples=125,
                         max_depth=7,
                         num_leaves=128,
                         reg_alpha=0.1,
                         reg_lambda=0.1,
                         # scale_pos_weight=83.7
                         )

    gbm.fit(X_train, y_train)

    joblib.dump(gbm, 'model_files/LGBM_model.pkl')
    gbm = joblib.load('model_files/LGBM_model.pkl')

    y_pred = gbm.predict_proba(x_test_all, num_iteration=gbm.best_iteration_)
    print(y_pred.shape)
    np.save("submit/output.npy", y_pred)


if __name__ == "__main__":
    main()
