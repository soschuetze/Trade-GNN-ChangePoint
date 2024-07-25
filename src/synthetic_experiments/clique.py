import argparse
import numpy as np
import networkx as nx
import torch
import pickle
import dgl
import os
import sys
from tqdm import trange
from os.path import join
import random
from sample import sample_pairs, sample_pygcn_graph
from pathlib import PosixPath
import json
from datetime import datetime


def generate_pairs_clique(args=None):

    sbm_1 = {'type': 'er', 'sizes': [args.n_nodes - args.size_clique, args.size_clique], 'p': [args.p, args.p], 'q': args.q, "features":args.features}
    sbm_2 = {'type': 'sbm','sizes': [args.n_nodes - args.size_clique, args.size_clique], 'p': [args.p, args.p], 'q': args.q, "features":args.features}

    data = []

    # negative label data points
    for _ in trange(args.n_samples, desc='Generating negative examples'):
        pair = [sample_pygcn_graph(sbm_1), sample_pygcn_graph(sbm_2)]
        random.shuffle(pair)
        graph1, graph2 = pair

        label = torch.tensor(0)
        data.append((graph1, graph2, label))

    # positive label data points
    for _ in trange(args.n_samples, desc='Generating positive examples'):
        if random.random() < 0.5:
            sbm = sbm_1
        else:
            sbm = sbm_2
        graph1, graph2 = sample_pygcn_graph(sbm), sample_pygcn_graph(sbm)
        label = torch.tensor(1)
        data.append((graph1, graph2, label))

    random.shuffle(data)

    # save data
    save_dir = args.save_dir + f"{2*args.n_samples}_pairs_clique_sbm_{args.n_nodes}_{args.size_clique}_{args.p}_{args.q}_{args.features}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(data, open(save_dir / 'data.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)


def generate_sequence_clique(args=None):
    """Generate a time series of graphs where the underlying SBM parameters change.

    sbm_1/sbm_2: parameter dictionaries for the SBM before and after the change point
    length:
    sizes: list containing the number of nodes in each community
    length: number of graphs in each block
    """

    cp_time = np.random.randint(args.n_samples // 4, 3 * args.n_samples // 4)
    #print(f" Change point at t = {cp_time}")
    num_nodes = args.n_nodes
    sbm_1 = {'num_nodes':num_nodes, 'type': 'er', 'sizes': [args.n_nodes - args.size_clique, args.size_clique], 'p': [args.p, args.p], 'q': args.q, "features":args.features}
    sbm_2 = {'num_nodes':num_nodes, 'type': 'sbm','sizes': [args.n_nodes - args.size_clique, args.size_clique], 'p': [args.p, args.p], 'q': args.q, "features":args.features}
    g1, g2 = [], []
    for _ in range(cp_time):
        g1.append(sample_pygcn_graph(sbm_1))
    for _ in range(args.n_samples- cp_time):
        g2.append(sample_pygcn_graph(sbm_2))
    seq = g1 + g2
    lab = [0] * cp_time + [1] * (args.n_samples- cp_time)

    return seq, cp_time, lab


def generate_sequence_clique_multiple(args=None):
    """

    Generate a dynamic network sequence with multiple change-points corresponding to appearance/disappearance of a clique

    sbm_1/sbm_2: parameter dictionaries for the SBM before and after the change point
    length:
    sizes: list containing the number of nodes in each community
    length: number of graphs in each block
    """

    dynnet = []
    cp_times = []
    labels = []
    for k in range(args.n_changes):
        seq, cp, lab = generate_sequence_clique(args)
        dynnet = dynnet + seq
        cp_times = cp_times + [k*args.n_samples + cp, (k+1)*args.n_samples]
        labels = labels + list(np.array(lab) * (k+1))
    cp_times.pop()

    print(f" Change points at t = {cp_times}")

    # save data
    time_exp = str(datetime.utcnow().strftime("%m_%d_%H:%M:%S"))
    save_dir = args.save_dir +  f"{time_exp}_clique_cp_{args.n_changes}_T_{args.n_changes*args.n_samples}_n_{args.n_nodes}_p_{args.p}_q_{args.q}_{args.size_clique}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(dynnet, open(save_dir / 'data.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    # save change point time
    with open(save_dir / 'time.json', 'w') as fp:
        json.dump(cp_times, fp)

    # save labels
    with open(save_dir / 'labels.p', 'wb') as fp:
        pickle.dump(labels, fp)

    return dynnet, cp_times, labels, str(save_dir)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, default=400)
    parser.add_argument('--n_changes', type=int, default=5)
    parser.add_argument('--size_clique', type=int, default=80)
    parser.add_argument('--p', type=float, default=0.08)
    parser.add_argument('--q', type=float, default=0.02)
    parser.add_argument('--n_samples', type=int, default=480)
    parser.add_argument('--features', type=str, choices=['gaussian'], default=None)
    parser.add_argument('--sequence', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str,  default='results/synthetic/')
    parser.add_argument('--rep', type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if not args.sequence:
        generate_pairs_clique(args=args)
    else:
        generate_sequence_clique_multiple(args=args)

if __name__ == '__main__':
    main()