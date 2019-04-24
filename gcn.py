"""
GCN implementation of the DGL library (https://dgl.ai) with minor modifications
to facilitate dynamically changing graph structure.

Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn_mp.py
"""
import math
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


def gcn_norm(g):
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    return norm


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = None
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

        # Compute initial norm
        self.set_graph(g)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, self.g)
        return h

    def set_graph(self, new_graph):
        norm = gcn_norm(new_graph)
        # if use_cuda:
        #     norm = norm.cuda()
        new_graph.ndata['norm'] = norm.unsqueeze(1)

        self.g = new_graph
        return self
