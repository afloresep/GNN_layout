import numpy as np
import scipy.sparse as sp
import torch

def compute_lpe(edge_index, num_nodes, k):
    # Compute the graph Laplacian
    edge_index = edge_index.t().cpu().numpy()
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.array(adj.sum(1)).flatten()
    D = sp.diags(deg)
    L = D - adj

    # Compute eigenvectors
    eigval, eigvec = np.linalg.eig(L.toarray())
    idx = eigval.argsort()[1:k+1]  # exclude the first (zero) eigenvalue
    return torch.from_numpy(eigvec[:, idx]).float()


"""
This implementation computes the k smallest non-trivial eigenvectors of the graph Laplacian. 
If k is larger than the number of available eigenvectors, it will automatically be limited to the maximum number available.
"""