import random
import time

import numpy as np
import torch
import torch.nn as nn
from comet_ml import Experiment
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

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

    accuracy = 0
    times = []
    scores = []

    for f in range(param["num_folds"]):
        time_start = time.time()

        optimizer.zero_grad()

        loss = model.loss(data.train_data[f], data.train_labels[f])

        loss.backward()

        optimizer.step()

        time_end = time.time()

        times.append(time_end - time_start)

        val_out = model.forward(data.valid_data[f])

        scores.append(
            precision_recall_fscore_support(data.valid_labels[f].data.numpy(), val_out.data.numpy().argmax(axis=1)))

        accuracy += loss.data[0]

        print(f, loss.data[0],
              f1_score(data.valid_labels[f].data.numpy(), val_out.data.numpy().argmax(axis=1), average="micro"))

    # classify test vertices
    test_out = model.forward(data.test_data)

    print(1.0 * accuracy / param["num_folds"])
    print_report(scores, param["num_classes"], param["num_folds"])
    print("Average batch training time:", np.mean(times))
    print(">> Test Evaluation")
    print("F1 Score:", f1_score(data.test_labels.data.numpy(), test_out.data.numpy().argmax(axis=1), average="micro"))
    print("Confusion Matrix\n", confusion_matrix(data.test_labels.data.numpy(), test_out.data.numpy().argmax(axis=1)))


def print_report(scores, num_classes, num_folds):
    precision = recall = f1 = support = [0] * num_classes

    for score in scores:
        for c in range(num_classes):
            precision[c] = score[0][c]
            recall[c] = score[1][c]
            f1[c] = score[2][c]
            support[c] = score[3][c]

    for c in range(num_classes):
        print("Class %d averages [p: %f, r: %f, f1: %f, s: %f]" % (c, precision[c]/num_folds, recall[c]/num_folds, f1[c]/num_folds, support[c]/num_folds))


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

    param_citeseer = {
        "num_classes": 6,
        "num_nodes": 3312,
        "num_features": 3703,
        "num_folds": 100,
        "dim1": 128,
        "dim2": 128,
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.4,
        "lr_decay": 0.015
    }

    param_pubmed = {
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 100,
        "dim1": 128,
        "dim2": 128,
        "sample1": 15,
        "sample2": 10,
        "learning_rate": 0.5,
        "lr_decay": 0.005
    }

    run(param_cora, Data.load_cora)

    # run(param_citeseer, Data.load_citeseer)

    # run(param_pubmed, Data.load_pubmed)
