import random
import time

import numpy as np
import torch
import torch.nn as nn
from comet_ml import Experiment
from sklearn.metrics import f1_score
from torch.autograd import Variable

from graphsage.data import Cora
from graphsage.model import SupervisedGraphSage
from graphsage.layers import Encoder, MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = Cora.load_cora()

    # 2708 papers in dataset, 1433 (vocab of unique words) dimensional embeddings
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # layer 1
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(adj_lists, agg1, 1433, 128, 10)

    # layer 2
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t())
    enc2 = Encoder(adj_lists, agg2, enc1.embed_dim, 128, 10, base_model=enc1)

    model = SupervisedGraphSage(7, enc2)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = model.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.data[0])

    val_output = model.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


# def run_pubmed():
#     np.random.seed(1)
#     random.seed(1)
#     num_nodes = 19717
#     feat_data, labels, adj_lists = load_pubmed()
#
#     features = nn.Embedding(19717, 500)
#     features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
#
#     agg1 = MeanAggregator(features, cuda=True)
#     enc1 = Encoder(features, 500, 128, adj_lists, agg1, 10)
#
#     agg2 = MeanAggregator(lambda nodes: enc1(nodes).t())
#     enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2, 25, base_model=enc1)
#
#     graphsage = SupervisedGraphSage(3, enc2)
#
#     rand_indices = np.random.permutation(num_nodes)
#     test = rand_indices[:1000]
#     val = rand_indices[1000:1500]
#     train = list(rand_indices[1500:])
#
#     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
#     times = []
#     for batch in range(200):
#         batch_nodes = train[:1024]
#         random.shuffle(train)
#         start_time = time.time()
#         optimizer.zero_grad()
#         loss = graphsage.loss(batch_nodes,
#                               Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
#         loss.backward()
#         optimizer.step()
#         end_time = time.time()
#         times.append(end_time - start_time)
#         print(batch, loss.data[0])
#
#     val_output = graphsage.forward(val)
#     print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
#     print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    experiment = Experiment(api_key="T89lpyGziH2MDRAfdJ0G0LpSr", project_name="inductivegcn")
    run_cora()
