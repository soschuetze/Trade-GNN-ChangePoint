import networkx as nx
import numpy as np
import os
import pickle
import argparse
import json
from torch_geometric.utils import to_networkx
from sample import sample_pairs
import torch

def get_graph_pairs(args=None):
    # Assuming root_dir is the path to your root directory
    args_dict = vars(args)  # arguments as dictionary
    root_dir = args_dict['root_dir']

    clique_data = {}
    cp_times = {}
    label_data = {}

    # Walk through all directories and files in root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If there's a data.p file in this directory, read it
        args_file = os.path.join(dirpath, 'args.json')
        if os.path.isfile(args_file):
            with open(args_file, 'rb') as f:
                arg_data = json.load(f)
                clique_size = arg_data['size_clique']

        data_file = os.path.join(dirpath, 'data.p')
        if os.path.isfile(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                clique_data[clique_size] = data

        # If there's a time.json file in this directory, read it
        time_file = os.path.join(dirpath, 'time.json')
        if os.path.isfile(time_file):
            with open(time_file, 'r') as f:
                time_data = json.load(f)
                cp_times[clique_size] = time_data

        label_file = os.path.join(dirpath, 'labels.p')
        if os.path.isfile(label_file):
            with open(label_file, 'rb') as f:
                data = pickle.load(f)
                label_data[clique_size] = data

    s = args_dict['clique_size']
    for j, i in enumerate(clique_data[s]):
        edge_index = i.edge_index.to(torch.int64)
        networkx_graph = to_networkx(i)
        adjacency = nx.adjacency_matrix(networkx_graph)
        
        attributes = np.eye(adjacency.shape[0])
        clique_data[s][j].x = attributes
    
    train = clique_data[s][:1000]
    train_labels = label_data[s][:1000]

    val = clique_data[s][1000:2000]
    val_labels = label_data[s][1000:2000]

    test = clique_data[s][2000:]
    test_labels = label_data[s][2000:]

    graph_pairs_train = sample_pairs(train,train_labels,nsamples=2000)
    graph_pairs_val = sample_pairs(val,val_labels,nsamples=1000)

    time_test = [t-2000 for t in cp_times[s] if t>=2000]

    with open(f'../../results/test_synthetic/test/{s}-data.p', 'wb') as f:
        pickle.dump(test, f)

    with open(f'../../results/test_synthetic/test/{s}-labels.p', 'wb') as f:
        pickle.dump(test_labels, f)

    with open(f'../../results/test_synthetic/test/{s}-time.json', 'w') as f:
        json.dump(time_test, f)

    with open(f'../../synthetic/graph_pairs/graph_pairs_train_{s}.p', 'wb') as f:
        pickle.dump(graph_pairs_train, f)
    with open(f'../../synthetic/graph_pairs/graph_pairs_val_{s}.p', 'wb') as f:
        pickle.dump(graph_pairs_val, f)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../synthetic_experiments/results/synthetic', help='Path to root directory.')
    parser.add_argument('--clique_size', type=int, default=80, help='S value.')


    args = parser.parse_args()

    return args

def main():
    args = get_args()
    args_dict = vars(args)  # arguments as dictionary

    get_graph_pairs(args = args)

if __name__ == '__main__':
    main()