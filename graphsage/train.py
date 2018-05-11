import random
import time
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import graphsage.sampler as sampler
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

from graphsage.aggregator import MeanAggregator
from graphsage.data import Data
from graphsage.layers import Encoder
from graphsage.model import SupervisedGraphSage
from graphsage.param import Param

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
    sam1 = sampler.get(param["sampler1"], param["sample1"], data)
    agg1 = MeanAggregator(features, sam1, cuda=True)
    enc1 = Encoder(data.adj_lists, agg1, features.embedding_dim, param["dim1"])

    # layer 2
    sam2 = sampler.get(param["sampler2"], param["sample2"], data)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), sam2)
    enc2 = Encoder(data.adj_lists, agg2, enc1.embedding_dim, param["dim2"], base_model=enc1)

    model = SupervisedGraphSage(param["num_classes"], enc2)

    optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=param["learning_rate"],
                                 lambd=param["lr_decay"])

    times_fold = []
    times_batch = []
    scores = []
    batch_size = param["batch_size"]
    num_batches = ceil(len(data.train_labels[0]) / batch_size)

    for f in range(param["num_folds"]):
        f_time_start = time.time()

        optimizer.zero_grad()

        batch = 1
        while len(data.train_labels[f]) > 0:
            print("Starting batch {0} of {1}\r".format(batch, num_batches), end="")

            b_time_start = time.time()

            loss = model.loss(data.train_data[f][:batch_size], data.train_labels[f][:batch_size])

            loss.backward()

            optimizer.step()

            if len(data.train_data[f]) < batch_size:
                break

            data.train_data[f] = data.train_data[f][batch_size:]

            data.train_labels[f] = data.train_labels[f][batch_size:]

            b_time_end = time.time()

            times_batch.append(b_time_end - b_time_start)

            batch = batch + 1

        f_time_end = time.time()

        times_fold.append(f_time_end - f_time_start)

        val_out = model.forward(data.valid_data[f])

        scores.append(
            precision_recall_fscore_support(data.valid_labels[f].data.numpy(), val_out.data.numpy().argmax(axis=1)))

        print("Epoch", f, f1_score(data.valid_labels[f].data.numpy(), val_out.data.numpy().argmax(axis=1), average="micro"))

    # classify test vertices
    test_out = model.forward(data.test_data)

    # print_report(scores, param["num_classes"], param["num_folds"])
    print("Average epoch-training time:", np.mean(times_fold))
    print("Average batch-training time:", np.mean(times_batch))
    print(">> Test Evaluation")
    print("F1 Score:", f1_score(data.test_labels.data.numpy(), test_out.data.numpy().argmax(axis=1), average="micro"))
    print("Confusion Matrix\n", confusion_matrix(data.test_labels.data.numpy(), test_out.data.numpy().argmax(axis=1)))


if __name__ == "__main__":
    # disabling comet.ml
    # experiment = Experiment(api_key="T89lpyGziH2MDRAfdJ0G0LpSr", project_name="scalablegcn")

    #run(Param.cora_mixed, Data.load_cora)
    #run(Param.cora_rand, Data.load_cora)
    run(Param.cora_priority, Data.load_cora)
    #run(Param.cora_hybd, Data.load_cora)
    #run(Param.citeseer, Data.load_citeseer)
    #run(Param.citeseer_rand, Data.load_citeseer)
    #run(Param.citeseer_imp, Data.load_citeseer)
    run(Param.citeseer_hybd, Data.load_citeseer)
    #run(Param.citeseer_mixed, Data.load_citeseer)
    #run(Param.pubmed_imp, Data.load_pubmed)
    #run(Param.pubmed_rand, Data.load_pubmed)
    #run(Param.pubmed_mixed, Data.load_pubmed)
    run(Param.pubmed_hybd, Data.load_pubmed)
