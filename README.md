`Dataset (in data/dataset.py)`
Handles loading and preprocessing of custom graph dataset


`LaplacianPE (in data/laplacian_pe.py)`
Computes Laplacian Positional Encoding for input graphs


`GCN (in models/gcn.py)`
Implements Graph Convolutional Network layers


`LatentGNNKernel (in models/latent_gnn.py)`
Implements a single LatentGNN kernel


`LayoutModel (in models/layout_model.py)`
Main model class that combines all components


`LayoutLoss (in utils/loss.py)`
Implements the custom loss function described in the paper


`EnergyMetric (in utils/metrics.py)`
Implements the evaluation metric (energy relative difference)

`SBMLayoutDataset (in data/sbm_dataset.py)`
A dataset of Stochastic Block Model (SBM) graphs with force-directed layouts.

This dataset extends PyTorch Geometric's StochasticBlockModelDataset by adding Laplacian Positional Encoding (LPE) as node features and generating force-directed layouts as ground truth for graph layout tasks.

- `num_channels`: This parameter is specific to the LPE. It dermines the dimensionality of the node features created by the LPE. It's the number of egienvectors (k) and thus the number of features per node use to enconde the graph structure 

- `edge_probs`: This is a parameter of the SBM. It's a matrix where `edge_probs[i][j]`represents the probability of an edge existing between node in block `i`and `j`. Readmore 


---
Example  of usage:


```python

dataset = SBMLayoutDataset(
    root='data/sbm',
    num_graphs=100,  # Generate 100 graphs
    num_nodes=600,   # Each graph has 600 nodes in total
    block_sizes=[100, 200, 300],  # Divided into 3 blocks of sizes 100, 200, and 300
    edge_probs=[
        [0.5, 0.1, 0.1],
        [0.1, 0.5, 0.1],
        [0.1, 0.1, 0.5]
    ],
    num_channels=40  # Use 40 dimensions for the Laplacian Positional Encoding
)
```

This will create a dataset of 100 graphs, each with 600 nodes divided into 3 blocks. Nodes within the same block have a higher probability of being connected (0.5) compared to nodes in different blocks (0.1). Each node will be represented by a 40-dimensional feature vector derived from the Laplacian Positional Encoding.



---
### Eigenvectors and Eigenvalues

For a more detailed explanation of the role of eigenvectors, eigenvalues and LPE on the 2D layout of a graph, please check out my [blog](https://afloresep.github.io/posts/2024/10/laplacian_positional_encoding/)
