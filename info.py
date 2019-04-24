""" Print dataset info with subgraph train-test split """
import argparse
import torch
import dgl
from dgl.data import register_data_args, load_data

def main(args):
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

    udgraph = data.graph

    g = dgl.DGLGraph(udgraph)

    self_edges = g.has_edges_between(g.nodes(), g.nodes()).sum()
    print("Orig data has %d self edges" % self_edges)

    n_edges = g.number_of_edges() - self_edges  # Don't count self edges here

    src, dst = g.all_edges()
    is_symmetric = all(g.has_edges_between(dst, src))
    print("Is symmetric:", is_symmetric)
    if not is_symmetric:
        print("WARN the input graph is non-symmetric")


    


    # g.add_edges(g.nodes(), g.nodes())

    # assert all(g.has_edges_between(g.nodes(), g.nodes()))

    train_nodes = torch.arange(g.number_of_nodes())[train_mask]

    g_train = g.subgraph(train_nodes)
    # assert all(g_train.has_edges_between(g_train.nodes(), g_train.nodes()))

    assert g != g_train



    g_train.set_n_initializer(dgl.init.zero_initializer)
    features_train = features[train_mask]
    labels_train = labels[train_mask]

    self_edges_train = g_train.has_edges_between(g_train.nodes(), g_train.nodes()).sum().item()
    print("Self edges in train set", self_edges_train)
    n_edges_train = g_train.number_of_edges() - self_edges_train

    unseen_nodes = g.number_of_nodes() - g_train.number_of_nodes()
    # Only real unseen edges, not the included train edges
    unseen_edges = n_edges - n_edges_train


    print("""---- Data statistics: %s, Setting %s----
      #Full Graph Nodes %d 
      #Full Graph Edges %d (undirected: %d)
      #Classes %d
      #Features %d
      #Train samples %d
      #Train edges %d (undirected: %d)
      #Unseen nodes %d
      #Unseen edges %d (undirected: %d)
      #Test nodes %d
      #Label rate %.3f""" %
          (args.dataset,
           'B' if args.invert else 'A',
           g.number_of_nodes(), n_edges, n_edges // 2,
           n_classes, in_feats,
           g_train.number_of_nodes(),
           n_edges_train, n_edges_train // 2,
           unseen_nodes,
           unseen_edges, unseen_edges // 2,
           test_mask.sum().item(),
           g_train.number_of_nodes() / g.number_of_nodes()
          )
         )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    register_data_args(parser)
    parser.add_argument('--invert', default=False, action='store_true',
                        help="Invert train and test set")
    args = parser.parse_args()
    main(args)

