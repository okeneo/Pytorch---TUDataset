import random

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset

"""Need these installed to use torch_geometric:
pip install torch_sparse # This will take some time.
pip install torch_scatter # This will take some time.
pip install torch_geometric
"""

"""Information needed for each TUDataset:
Edge information i.e., which 2 nodes are adjacent? -> Tensors
Graph indicators i.e., what graph does each node/edge come from? -> Directly from dataset[index]
Graph labels/class i.e., what graph class does each of our graphs belong to? -> dataset[index].x
"""

# Allow for reproducibility.
random.seed(42)

TRAIN_PERCENT = 0.8
TEST_PERCENT = 0.2


def getGraphTupleList(each_graph):
    """Accepts a graph from a dataset (in it's TUDataset format) and converts it to a TupleList
    graph representation."""
    graph_edges = pd.DataFrame(columns=["from", "to"])

    # The node/edge information is stored in 2D Tensors.
    # The first row determines the "from" nodes, and the second row determines the "to" nodes.
    graph_edges["from"] = each_graph.edge_index[0, :]
    graph_edges["to"] = each_graph.edge_index[1, :]

    create_graph = ig.Graph.TupleList(
        graph_edges.itertuples(index=False), directed=False, weights=True
    )
    return create_graph


def getGraphLabel(dataset):
    """Accepts a TUDataset and returns a list of the graph labels/classes of each graph in the
    dataset from graph 1 to N (where N is the Nth graph in the dataset)"""
    graph_label = []
    for each_graph in dataset:
        # each_graph.y represents the graph class of each_graph.
        graph_label.append(each_graph.y.item())
    return graph_label


def read_dataset(dataset):
    """Accepts a TUDataset name and loads the dataset, splitting the graphs into 2 subsets
    (test and train) along with their corresponding graph labels."""
    dataset = TUDataset(root="/tmp/" + dataset + "/", name=dataset)  # Load graph.
    print("{}\n{}\n".format(dataset.name, dataset.data))

    graph_labels = getGraphLabel(dataset)
    unique_graph_indicator = np.arange(
        start=1, stop=len(dataset) + 1, step=1
    )  # a list from 1 to N (N - number of graphs in the dataset).
    x_train, x_test, y_train, y_test = train_test_split(
        unique_graph_indicator, graph_labels, test_size=TEST_PERCENT, random_state=42
    )
    return x_train, x_test, y_train, y_test, dataset


def main(dataset):
    x_train, x_test, y_train, y_test, dataset = read_dataset(dataset)
    # trainModel(x_train, y_train, dataset)
    # testModel(x_test, y_test, dataset)


if __name__ == "__main__":
    # List of datasets.
    DATASET_LIST = (
        "BZR",
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "DHFR",
        "NCI1",
        "COX2",
    )
    for dataset in DATASET_LIST:
        main(dataset)
