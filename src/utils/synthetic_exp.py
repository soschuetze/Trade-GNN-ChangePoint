import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import os
import pickle
import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path, PosixPath
import json
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, roc_auc_score, f1_score

from sample import sample_pairs
from misc import collate
from src.model import GraphSiamese
from embedding import GCN

def train(args=None):

    args_dict = vars(args)

    topk = args_dict['top_k']
    s = args_dict['clique_size']
    batch_size = args_dict['batch_size']

    with open(f'../../synthetic/graph_pairs/graph_pairs_train_{s}.p', 'rb') as f:
        graph_pairs_train = pickle.load(f)
    with open(f'../../synthetic/graph_pairs/graph_pairs_val_{s}.p', 'rb') as f:
        graph_pairs_val = pickle.load(f)
    
    training_data_pairs = DataLoader(graph_pairs_train, batch_size=batch_size, shuffle=True, collate_fn=collate,
                               drop_last=True)
    validation_data_pairs = DataLoader(graph_pairs_val, batch_size=batch_size, shuffle=True, collate_fn=collate,
                               drop_last=True)

    input_dim = training_data_pairs.dataset[0][0].x.shape[1]*batch_size
    embedding = GCN(input_dim=input_dim, type='gcn', hidden_dim=16, layers=args_dict['nlayers'], dropout=args_dict['dropout'])
    model = GraphSiamese(embedding, args_dict['distance'], args_dict['pooling'], args_dict['loss'], topk, nlinear=args_dict['nlayers_mlp'],
                            nhidden=16, dropout=args_dict['dropout'], features=None)
    
    optimizer = optim.Adam(model.parameters(), lr=args_dict['lr'], weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    loss_fn = torch.nn.BCELoss(reduction='none')

    logging = {'train_loss': [], 'train_acc': [], 'train_recall': [], 'train_precision': [],
                   'valid_loss': [], 'valid_acc': [], 'valid_precision': [], 'valid_recall': []}

    best_f1, best_weights, best_loss = 0., None, np.Inf
    final_metrics = {'loss' : [0.0, 0., 0.], 'accuracy': [0., 0., 0.], 'recall': [0., 0., 0.], 'precision': [0., 0., 0.]}

    # for early stopping
    patience = 10
    patience_counter = 0

    # training loop
    for epoch in range(args_dict['nepochs']):

        # training updates
        train_loss, train_acc, train_precision, train_recall = [], [], [], []

        model.train()

        # minibatch loop
        for (graph1, graph2, labels) in training_data_pairs:
            
            graph1, graph2, labels = graph1, graph2, labels

            predictions = model(graph1, graph2)
            predictions = torch.sigmoid(predictions) # predictions between 0 and 1

            loss = loss_fn(predictions, labels.float())

            # balanced accuracy score instead of plain accuracy
            accuracy = torch.tensor(np.array((predictions.squeeze().cpu().detach() > 0.5) == labels.cpu(),
                                                dtype=float).mean().item()).unsqueeze(dim=0)
            recall = torch.tensor(
                [recall_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])
            precision = torch.tensor(
                [precision_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])

            train_loss.append(loss), train_acc.append(accuracy), train_recall.append(recall), train_precision.append(precision), \

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


        logging['train_loss'].append(torch.cat(train_loss).mean().item())
        logging['train_acc'].append(torch.cat(train_acc).mean().item())
        logging['train_recall'].append(torch.cat(train_recall).mean().item())
        logging['train_precision'].append(torch.cat(train_precision).mean().item())

        scheduler.step()

        # validation updates
        model.eval()

        valid_loss, valid_acc, valid_recall, valid_precision = [], [], [], []
        with torch.no_grad():
            for (graph1, graph2, labels) in validation_data_pairs:

                graph1, graph2 = graph1, graph2

                predictions = model(graph1, graph2)

                predictions = torch.sigmoid(predictions)

                loss = loss_fn(predictions, labels.float())

                recall = torch.tensor(
                    [recall_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
                precision = torch.tensor(
                    [precision_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
                accuracy = torch.tensor(np.array((predictions.squeeze().detach().cpu() > 0.5).float() == labels.float(),
                                                    dtype=float).mean().item()).unsqueeze(dim=0)

                valid_loss.append(loss), valid_acc.append(accuracy), valid_recall.append(
                    recall), valid_precision.append(
                    precision)

            logging['valid_loss'].append(torch.cat(valid_loss).mean().item())
            logging['valid_acc'].append(torch.cat(valid_acc).mean().item())
            logging['valid_recall'].append(torch.cat(valid_recall).mean().item())
            logging['valid_precision'].append(torch.cat(valid_precision).mean().item())

            # save best weights
            #if logging['valid_f1'][-1] > best_f1 and epoch > 0:
            if logging['valid_loss'][-1] < best_loss and epoch > 0:
                best_loss = logging['valid_loss'][-1]
                #best_f1 = logging['valid_f1'][-1]
                final_metrics['loss'][:2] = [logging['train_loss'][-1], logging['valid_loss'][-1]]
                final_metrics['accuracy'][:2] = [logging['train_acc'][-1], logging['valid_acc'][-1]]
                final_metrics['recall'][:2] = [logging['train_recall'][-1], logging['valid_recall'][-1]]
                final_metrics['precision'][:2] = [logging['train_precision'][-1], logging['valid_precision'][-1]]
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1


        if patience == patience_counter:
            break

        if epoch % 1 == 0:

            train_acc, train_loss = logging['train_acc'][-1], logging['train_loss'][-1]

            valid_acc, valid_loss = logging['valid_acc'][-1], logging['valid_loss'][-1]
            print("Epoch, Training loss, Valid loss, Valid Acc", epoch, train_loss, valid_loss,
                    valid_acc)
            print("Patience counter : ", patience_counter)

    model_path = (f's_{s}_k_{topk}')

    save_dir = PosixPath('../../synthetic/trained_models/').expanduser() / model_path
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pd.DataFrame(logging).to_csv(save_dir / 'logging.csv')

    torch.save(best_weights, save_dir /'model.pt')

    with open(save_dir / 'results.json', 'w') as fp:
        json.dump(final_metrics, fp, indent=2)

    return str(save_dir)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clique_size', type=int, default=60)
    parser.add_argument('--training_data', type=str, default=None)
    parser.add_argument('--validation_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--path_to_hps', type=str, default=None, help='Path to hyperparameters.')
    parser.add_argument('--validation_proportion', type=float, default=0.2)
    parser.add_argument('--test_proportion', type=float, default=0.2)
    parser.add_argument('--n_pairs', type=int, default=5000)
    parser.add_argument('--pair_sampling', type=str, default='random', choices=['random', 'window'])
    parser.add_argument('--window_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size during training.')
    parser.add_argument('--embedding_module', type=str, default='gcn', choices=['identity', 'gcn', 'gin', 'gat'],
                        help='Model to use for the node embedding.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers of the graph encoder.')
    parser.add_argument('--nlayers_mlp', type=int, default=2, help='Number of layers of the MLP following topk.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units in each layer of the graph encoder.')
    parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='How to compute distance after the embedding.')
    parser.add_argument('--pooling', type=str, default='topk', choices=['average', 'topk', 'avgraph', 'max'],
                        help='Pooling layer to use after computing similarity or distance.')
    parser.add_argument('--loss', type=str, default='bce', choices=['hinge', 'bce', 'mse', 'contrastive'],
                        help='Loss function to use on predictions.')
    parser.add_argument('--weight_loss', type=float, default=1.0,
                        help='Weight on negative examples in loss function.')
    parser.add_argument('--margin_loss', type=float, default=1.0,
                        help='Margin parameter in loss function.')
    parser.add_argument('--top_k', type=int, default=None, help='Number of nodes in top-k pooling.')
    parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--save_dir', type=str, default='~/PycharmProjects/GraphSiamese/trained_models/')
    parser.add_argument('--features', type=str, default=None, choices=['degree', 'random_walk', 'laplacian', 'identity'], help='Type of added input features')
    parser.add_argument('--input_dim', type=int, default=None, help='Dimension of input features if needed to be added')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Patience parameter for early stopping.')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    hparams = {}

    args_dict = vars(args)  # arguments as dictionary

    for key in hparams:
        if (key != 'dataset') and (key != 'save_dir') and (key != 'cuda') and (key != 'validation_dataset') :
            args_dict[key] = hparams[key]


    model_path = train(args=args)

if __name__ == '__main__':
    main()