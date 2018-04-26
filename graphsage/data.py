from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable


class Data(object):
    """
    Splits the input data into training and validation sets
    """
    def __init__(self, features, labels, num_nodes, num_folds):
        rand_indices = np.random.permutation(num_nodes)

        self.train_data = []
        self.train_labels = []

        self.valid_data = []
        self.valid_labels = []

        self.test_data = rand_indices[1000:]
        self.test_labels = Variable(torch.LongTensor(labels[np.array(self.test_data)]))

        ss = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.20, random_state=42)

        train_indices = rand_indices[:1000]
        for train_index, valid_index in ss.split(features[train_indices], labels[train_indices]):
            self.train_data.append(train_index)
            self.train_labels.append(Variable(torch.LongTensor(labels[np.array(train_index)])))

            self.valid_data.append(valid_index)
            self.valid_labels.append(Variable(torch.LongTensor(labels[np.array(valid_index)])))


class Cora:
    @staticmethod
    def load_cora():
        num_nodes = 2708
        num_feats = 1433
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes, 1), dtype=np.int64)
        node_map = {}
        label_map = {}
        with open("cora/cora.content") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data[i, :] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]

        adj_lists = defaultdict(set)
        with open("cora/cora.cites") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
        return feat_data, labels, adj_lists


# def load_pubmed():
#     # hardcoded for simplicity...
#     num_nodes = 19717
#     num_feats = 500
#     feat_data = np.zeros((num_nodes, num_feats))
#     labels = np.empty((num_nodes, 1), dtype=np.int64)
#     node_map = {}
#     with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
#         fp.readline()
#         feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
#         for i, line in enumerate(fp):
#             info = line.split("\t")
#             node_map[info[0]] = i
#             labels[i] = int(info[1].split("=")[1]) - 1
#             for word_info in info[2:-1]:
#                 word_info = word_info.split("=")
#                 feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
#     adj_lists = defaultdict(set)
#     with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
#         fp.readline()
#         fp.readline()
#         for line in fp:
#             info = line.strip().split("\t")
#             paper1 = node_map[info[1].split(":")[1]]
#             paper2 = node_map[info[-1].split(":")[1]]
#             adj_lists[paper1].add(paper2)
#             adj_lists[paper2].add(paper1)
#     return feat_data, labels, adj_lists
