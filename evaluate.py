import argparse
import math
import time
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


from factory import build_model, build_optimizer
from gcn import gcn_norm


def calc_accuracy(logits, true_labels):
    __max_vals, max_indices = torch.max(logits, 1)
    acc = (max_indices == true_labels).sum().float() / true_labels.size(0)
    return acc.item()


def eval_inference(epoch, net, features, labels, mask):
    """
    Evaluates the net on the (full) graph and calculates the
    loss and accuracy for `mask`, which is usually the test set mask
    """
    with torch.no_grad():
        net.eval()
        dev_logits = net(features)
        dev_logp = F.log_softmax(dev_logits, 1)
        dev_loss = F.nll_loss(dev_logp[mask], labels[mask])
    accuracy_score = calc_accuracy(dev_logp[mask], labels[mask])
    print("Epoch {:05d} | Test Loss {:.4f} | Test Accuracy {:.4f}"
          .format(epoch, dev_loss.detach().item(), accuracy_score))
    return accuracy_score


def train_epoch(epoch,
                net,
                optimizer,
                features,
                labels,
                train_mask=None):
    net.train()
    logits = net(features)

    if train_mask is not None:
        # If mask is given, only optimize on the respective labels
        task_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    else:
        # No mask is given, optimize on all
        task_loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    task_loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Task Loss: {:.4f}".format(epoch, task_loss.detach().item()))


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def main(args):
    column_headers = ["dataset",
                      "setting",
                      "model",
                      "pretraining",
                      "epoch",
                      "accuracy"]
    use_cuda = args.use_cuda and torch.cuda.is_available()
    print("Using CUDA:", use_cuda)

    results_df = pd.DataFrame(columns=column_headers)

    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels

    # We dont use a validation set
    train_mask = train_mask | val_mask

    if args.invert:
        # This is different from swapping train and test mask
        # because train | test not cover the whole dataset
        train_mask, test_mask = ~train_mask, train_mask
        setting = 'B'
    else:
        setting = 'A'

    g = dgl.DGLGraph(data.graph)
    # Suppress warning
    g.set_n_initializer(dgl.init.zero_initializer)
    # add self loop
    g.add_edges(g.nodes(), g.nodes())

    # g_train, g = split_graph(g, train_mask)
    # # Select train nodes..
    train_nodes = torch.arange(g.number_of_nodes())[train_mask]
    if use_cuda:
        features, labels = features.cuda(), labels.cuda()
        train_mask, test_mask = train_mask.cuda(), test_mask.cuda()
        train_nodes = train_nodes.cuda()

    # .. to induce subgraph
    g_train = g.subgraph(train_nodes)
    g_train.set_n_initializer(dgl.init.zero_initializer)
    features_train = features[train_mask]
    labels_train = labels[train_mask]

    # Verify sizes of train set
    assert int(train_mask.sum().item()) == features_train.size(0)\
        == labels_train.size(0) == g_train.number_of_nodes()


    # Random Restarts
    for __ in range(args.runs):
        # Init net
        net = build_model(args.model, args.dataset, g_train, in_feats, n_classes)
        if use_cuda:
            net = net.cuda()
        print(net)

        # Init optimizers
        # optimizer = torch.optim.Adam(net.parameters(),
        #                              **training_optimizer_params)
        optimizer = build_optimizer(net.parameters(),
                                    args.model,
                                    args.dataset,
                                    inference=False)
        print("Optimizer", optimizer)

        # Pre-training
        for epoch in range(args.epochs):
            train_epoch(epoch+1, net, optimizer,
                        features_train, labels_train,
                        train_mask=None  # Use all labels of the *train* subgraph
                        )

        print("=== INFERENCE ===")
        net.set_graph(g)
        # Eval without inference epochs
        accuracy_score = eval_inference(0, net, features, labels, test_mask)
        results_df = results_df.append(
            pd.DataFrame([[args.dataset, setting, args.model, args.epochs, 0, accuracy_score]],
                         columns=column_headers), ignore_index=True
        )

        # Fresh optimizer for up-training at inference time
        # optimizer = torch.optim.Adam(net.parameters(),
        #                              **inference_optimizer_params)
        del optimizer
        optimizer = build_optimizer(net.parameters(),
                                    args.model,
                                    args.dataset,
                                    inference=True)

        print("Fresh inference optimizer", optimizer)
        for i in range(args.inference):
            train_epoch(i+1, net, optimizer,
                        features, labels,
                        train_mask=train_mask)

            accuracy_score = eval_inference(i+1, net, features, labels, test_mask)
            results_df = results_df.append(
                pd.DataFrame([[args.dataset, setting, args.model, args.epochs, i+1, accuracy_score]],
                             columns=column_headers), ignore_index=True
            )
        del net
        del optimizer
        torch.cuda.empty_cache()  # don't leak here


    print(args)
    for i in range(args.inference + 1):
        # Print results to command line
        rbi = results_df[results_df['epoch'] == i]['accuracy']
        print("Avg accuracy over {} runs after {} inference epochs: {:.4f} ({:.4f})".format(args.runs, i, rbi.mean(), rbi.std()))

    if args.outfile is not None:
        # And store them to csv file
        appendDFToCSV_void(results_df, args.outfile, sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    register_data_args(parser)
    parser.add_argument('--model', type=str, help="Specify model")
    parser.add_argument('--runs', default=1, type=int,
                        help="Number of random reruns")
    parser.add_argument('--inference', default=0, type=int,
                        help="Number of inference epochs")
    parser.add_argument('--invert', default=False, action='store_true',
                        help="Invert train and test set")
    parser.add_argument('--outfile', default=None, type=str,
                        help="Dump Results to outfile")
    parser.add_argument('--epochs', default=200, type=int,
                        help="Number of training epochs")
    parser.add_argument('--no-cuda', dest='use_cuda', default=True,
                        action='store_false',
                        help="Force no cuda")
    args = parser.parse_args()
    print(args)
    main(args)
