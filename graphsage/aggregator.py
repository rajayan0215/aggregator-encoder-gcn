import random
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


class MaxPoolingAggregator(nn.Module):

    def forward(self, nodes):
        pass


class MeanAggregator(nn.Module):
    """
    Aggregates using mean of the neighbors embeddings
    """

    def __init__(self, features, priority_list, num_sample=10, cuda=False):
        """
        Initializes the aggregator for a specific graph.

        Args:
            features (map) : lookup table that maps LongTensor of node ids to FloatTensor of feature values.
            num_sample (int, optional):
            cuda (bool, optional): if ``True``, embedding matrix will be a GPU tensors.
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.num_samples = num_sample
        self.cuda = cuda
        self.priority_list = priority_list

    def forward(self, nodes, neighbours_full):
        """
        Aggregate the feature information in a neighbourhood sample, for a batch of nodes.

        It performs mean-based aggregator convolutional as a linear approximation of the localized spectral convolution
        in the GCN architecture of Kipf and Welling (2017)

        Args:
            nodes (Tensor) : input batch of nodes
            neighbours_full (Tensor) : 2-D tensor representing the neighbourhood of each node
            num_sample (int, optional) : number of neighbours to sample, if ``0`` all neighbours get sampled, defaults to 10
        """

        # draw uniformly samples of size ``num_sample`` from the neighbours set of each node
        neighbours_sample = self.non_uniform_sample(neighbours_full, self.num_samples)

        # build neighbourhoods by joining each node with its sampled neighbours
        neighbourhoods = [set(neighbourhood).union([nodes[i]]) for i, neighbourhood in enumerate(neighbours_sample)]

        # build a list of unique nodes from all neighbourhoods
        neighbours_all = list(map(np.long, set.union(*neighbourhoods)))

        # build a weight matrix to average out the contribution of each feature attribute
        weight = self._build(neighbourhoods, neighbours_all)

        if self.cuda:
            feat_neighbours = self.features(torch.LongTensor(neighbours_all).cuda())
        else:
            feat_neighbours = self.features(torch.LongTensor(neighbours_all))

        feat_self = weight.mm(feat_neighbours)

        return feat_self

    def _build(self, neighbourhoods, nodes):
        # create a lookup table that maps node to it's index in the ``node_list``
        index_map = {node: index for index, node in enumerate(nodes)}

        # initialize a matrix of zeroes with a row for each neighbourhood and columns for each unique node
        mask = Variable(torch.zeros(len(neighbourhoods), len(nodes)))

        # collect indices of all nodes in the collective neighbourhood
        column_ids = [index_map[neighbour] for neighbourhood in neighbourhoods for neighbour in neighbourhood]

        # incrementally assign a unique row id to all nodes in the same neighbourhood
        row_ids = [i for i in range(len(neighbourhoods)) for _ in range(len(neighbourhoods[i]))]

        # create the mask vector and mark nodes in each neighbourhood with 1s
        mask[row_ids, column_ids] = 1

        if self.cuda:
            mask = mask.cuda()

        # sum up to find the size of each neighbourhood
        num_neigh = mask.sum(1, keepdim=True)

        # take the element-wise mean
        mask = mask.div(num_neigh)

        return mask

    @staticmethod
    def uniform_sample(neighbourhoods, num_sample, black_list=[]):
        """
        Sampling without replacement

        Args:
            neighbourhoods () :
            num_sample (int) : Number of items to sample from the 2nd dimension
            black_list (set, optional) : items to exclude
        """
        # local function pointers as a speed hack
        _set = set
        _sample = random.sample

        result = []
        for i in range(len(neighbourhoods)):
            neighbourhood = _set(neighbourhoods[i]) - _set(black_list[i])
            result.append(_sample(neighbourhood, num_sample) if num_sample < len(neighbourhood) else neighbourhood)

        return result

    def non_uniform_sample(self, neighbourhoods, num_sample):
        """
        Sampling without replacement

        Args:
            neighbourhoods () :
            priority_list () :
            num_sample (int) : Number of items to sample from the 2nd dimension
        """

        dim = int(num_sample / 2)
        important_ones = [self.priority_sample(neighbourhood, dim, self.priority_list) for neighbourhood in neighbourhoods]
        random_ones = self.uniform_sample(neighbourhoods, dim, important_ones)
        return [list(important_ones[i]) + list(random_ones[i]) for i in range(len(neighbourhoods))]

    def priority_sample(self, neighbourhood, num_samples, priority_list=None):
        sample = []
        dim = min(num_samples, len(neighbourhood))

        for i in range(len(priority_list)):
            if priority_list[i] in neighbourhood:
                sample.append(priority_list[i])
                if len(sample) == dim:
                    return sample

        return sample
