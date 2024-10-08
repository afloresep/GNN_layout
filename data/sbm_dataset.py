import torch
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import numpy as np


class SBMLayoutDataset(StochasticBlockModelDataset):
    """
    A dataset of Stochastic Block Model (SBM) graphs with force-directed layouts.

    This dataset extends PyTorch Geometric's StochasticBlockModelDataset by adding
    Laplacian Positional Encoding (LPE) as node features and generating force-directed
    layouts as ground truth for graph layout tasks.

    Args:
        root (str): Root directory where the dataset should be saved.
        num_graphs (int): Number of graphs to generate.
        num_nodes (int): Number of nodes in each graph.
        block_sizes (list): List of integers specifying the size of each block.
        edge_probs (list): List of lists specifying edge probabilities between blocks.
        num_channels (int, optional): Number of channels for Laplacian Positional Encoding. 
                                      Defaults to 40.

    Attributes:
        num_channels (int): Number of channels used for Laplacian Positional Encoding.

    Each graph in the dataset has the following attributes:
        - x (Tensor): Node features (Laplacian Positional Encoding)
        - edge_index (LongTensor): Graph connectivity in COO format
        - y (Tensor): Ground truth layout coordinates

    Example:
        >>> dataset = SBMLayoutDataset(
        ...     root='data/sbm',
        ...     num_graphs=100,
        ...     num_nodes=1000,
        ...     block_sizes=[200, 300, 500],
        ...     edge_probs=[[0.3, 0.02, 0.02],
        ...                 [0.02, 0.3, 0.02],
        ...                 [0.02, 0.02, 0.3]]
        ... )
        >>> len(dataset)
        100
        >>> dataset[0].x.shape
        torch.Size([1000, 40])
        >>> dataset[0].y.shape
        torch.Size([1000, 2])
    """


    def __init__(self, root, num_graphs, num_nodes, block_sizes, edge_probs, num_channels=40):
        super().__init__(root, num_graphs, num_nodes, block_sizes, edge_probs)
        self.num_channels = num_channels
        self.process()

    def process(self):
        for i, data in enumerate(self.data):
            # Compute Laplacian Positional Encoding
            lpe = self.compute_lpe(data.edge_index, data.num_nodes)
            data.x = lpe

            # Generate force-directed layout
            layout = self.generate_layout(data.edge_index, data.num_nodes)
            data.y = layout

            self.data[i] = data

    def compute_lpe(self, edge_index, num_nodes):
        # TODO: Implement Laplacian Positional Encoding
        # Could also use the implementation from laplacian_pe.py
        pass

    def generate_layout(self, edge_index, num_nodes):
        G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes))
        layout = nx.spring_layout(G)
        layout_array = np.array([layout[i] for i in range(num_nodes)])
        return torch.from_numpy(layout_array).float()
