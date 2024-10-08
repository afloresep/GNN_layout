from data.sbm_dataset import SBMLayoutDataset

# Example usage
dataset = SBMLayoutDataset(
    root='data/sbm',
    num_graphs=100,
    num_nodes=1000,
    block_sizes=[200, 300, 500],
    edge_probs=[[0.3, 0.02, 0.02],
                [0.02, 0.3, 0.02],
                [0.02, 0.02, 0.3]]
)