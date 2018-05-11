import torch
import torch.nn as nn
from torch.nn import init


class AggEncGCN(nn.Module):
    """
    Simple supervised AEGCN model using CE loss
    """

    def __init__(self, num_classes, encoder):
        super(AggEncGCN, self).__init__()
        self.enc = encoder
        self.xent = nn.CrossEntropyLoss()

        # registered module parameters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, encoder.embedding_dim))

        # to break symmetry between hidden units of the same layer during backpropagation
        # initialize weights using the method described in He, K. et al. (2015) using normal distribution
        init.xavier_normal(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
