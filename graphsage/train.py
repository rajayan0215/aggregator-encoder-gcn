import random
import time

import numpy as np
import torch
import torch.nn as nn
from comet_ml import Experiment
from sklearn.metrics import f1_score

from graphsage.data import Data
from graphsage.layers import Encoder
from graphsage.aggregator import MeanAggregator
from graphsage.model import SupervisedGraphSage

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


def run(param, data_loader):
    np.random.seed(1)
    random.seed(1)

    data = Data(data_loader, param["num_nodes"], param["num_folds"])

    # 2708 papers in dataset, 1433 (vocab of unique words) dimensional embeddings
    features = nn.Embedding(param["num_nodes"], param["num_features"])
    features.weight = nn.Parameter(torch.FloatTensor(data.features), requires_grad=False)

    # layer 1
    agg1 = MeanAggregator(features, data.priority_list, param["sample1"], cuda=True)
    enc1 = Encoder(data.adj_lists, agg1, features.embedding_dim, param["dim1"])

    # layer 2
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), data.priority_list, param["sample2"])
    enc2 = Encoder(data.adj_lists, agg2, enc1.embedding_dim, param["dim2"], base_model=enc1)

    model = SupervisedGraphSage(param["num_classes"], enc2)

    optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=param["learning_rate"],
                                 lambd=param["lr_decay"])

    times = []

    for f in range(param["num_folds"]):
        time_start = time.time()

        optimizer.zero_grad()

        loss = model.loss(data.train_data[f], data.train_labels[f])

        loss.backward()

        optimizer.step()

        time_end = time.time()

        times.append(time_end - time_start)

        val_out = model.forward(data.valid_data[f])

        print(f, loss.data[0],
              f1_score(data.valid_labels[f].data.numpy(), val_out.data.numpy().argmax(axis=1), average="micro"))

    test_out = model.forward(data.test_data)
    print("Test F1:", f1_score(data.test_labels.data.numpy(), test_out.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    experiment = Experiment(api_key="T89lpyGziH2MDRAfdJ0G0LpSr", project_name="inductivegcn")

    param_cora = {
        "num_classes": 7,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_folds": 100,
        "dim1": 128,
        "dim2": 128,
        "sample1": 7,
        "sample2": 4,
        "learning_rate": 0.5,
        "lr_decay": 0.005
    }

    param_pubmed = {
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 100,
        "dim1": 128,
        "dim2": 128,
        "sample1": 10,
        "sample2": 25,
        "learning_rate": 0.8,
        "lr_decay": 0.005
    }

    run(param_cora, Data.load_cora)

    # run(param_pubmed, Data.load_pubmed)
