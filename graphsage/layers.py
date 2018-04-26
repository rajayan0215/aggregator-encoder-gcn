import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
Collection of Sampling, Aggregation and Embedding layers
"""


class Encoder(nn.Module):
    """
    Encodes a node using the aggregate feature information from it’s local neighborhood.
    """

    def __init__(self, adj_lists, aggregator, feature_dim, embedding_dim, base_model=None, cuda=False):
        """
        Initializes the encoder for a specific graph.

        Args:
            adj_list ():
            aggregator ():
            feature_dim ():
            embedding_dim ():
            num_samples (int, optional): size of the neighbourhood, defaults to 10
            cuda (bool, optional): if ``True``, encoding uses GPU tensors.
        """

        super(Encoder, self).__init__()

        self.adj_lists = adj_lists

        self.aggregator = aggregator
        self.aggregator.cuda = cuda
        self.cuda = cuda

        self.feat_dim = feature_dim
        self.embedding_dim = embedding_dim

        if base_model is not None:
            self.base_model = base_model

        # registered module parameters
        self.weight = nn.Parameter(torch.FloatTensor(embedding_dim, self.feat_dim))

        # to break symmetry between hidden units of the same layer during backpropagation
        # initialize weights using Glorot and Bengio (2010) fan-in and fan-out method
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes for the next iteration.

        Args:
            nodes (Tensor): list of nodes
        """

        # collect aggregated neighborhood vector for each node
        feat_agg_neighbour = self.aggregator.forward(nodes, [self.adj_lists[node] for node in nodes])

        # apply nonlinear activation function with added randomness
        feat_self = F.rrelu(self.weight.mm(feat_agg_neighbour.t()))

        return feat_self

