import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.utils.data import DataLoader
def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                datasets: list[DataLoader], vector: Tensor):
    """Compute a Hessian-vector product."""

    p = len(parameters_to_vector(network.parameters()))
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()    
    # Making it compatible for global Hessian eigenvalue and SGD
    # even though most calls have 1 client and are full-batch
    n = 0
    for dataset in datasets:
        for X, y in dataset:
            network = network.cuda()
            loss = loss_fn(network(X.cuda()), y.cuda())  
            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)   
            dot = parameters_to_vector(grads).mul(vector).sum()
            grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
            hvp += parameters_to_vector(grads)
            n += 1
                       
    return hvp / n

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, datasets: list[DataLoader], neigs=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, datasets,
                                          delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals