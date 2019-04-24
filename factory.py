import torch
import torch.nn.functional as F

from gat import GAT
from gcn import GCN
from mlp import MLP
from graphsage import GraphSAGE

def build_model(model_key, dataset, g, in_feats, n_classes):
    """
    Returns a model instance based on --model command-line arg and dataset
    """
    if model_key == 'MLP':
        return MLP(in_feats, 64, n_classes, 1, F.relu, 0.5)
    elif model_key == 'GCN':
        return GCN(g, in_feats, 16, n_classes, 1, F.relu, 0.5)
    elif model_key == 'GCN-64':
        return GCN(g, in_feats, 64, n_classes, 1, F.relu, 0.5)
    elif model_key == 'GAT':
        # Default args from paper
        num_heads = 8
        num_out_heads = 8 if dataset == 'pubmed' else 1
        num_layers = 1  # one *hidden* layer
        heads = ([num_heads] * num_layers) + [num_out_heads]
        return GAT(g,
                   num_layers,
                   in_feats,
                   8,  # hidden units per layer
                   n_classes,
                   heads,
                   F.elu,  # activation fun
                   0.6,  # feat dropout
                   0.6,  # attn dropout
                   0.2,  # negative slope for leakyrelu
                   False  # Use residual connections
                   )
    elif model_key == 'GraphSAGE':
        return GraphSAGE(g, in_feats, 16, n_classes, 1, F.relu, 0.5, "mean")

    # Add more models here
    raise ValueError("Invalid model key")

def build_optimizer(parameters, model_key, dataset, inference=False):
    """
    Returns an optimizer instance based on --model command-line arg and dataset
    """
    # Inference is currently, as we use same optimizer params for inference as
    # we do for training
    if model_key == 'GAT':
        lr = 0.01 if dataset == 'pubmed' else 0.005
        weight_decay = 0.001 if dataset == 'pubmed' else 0.0005
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif model_key == 'GraphSAGE':
        lr = 1e-2
        wd = 5e-4
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)

    # Default optimizer (used for GCNs and MLPs)
    lr = 0.005
    weight_decay = 5e-4
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
