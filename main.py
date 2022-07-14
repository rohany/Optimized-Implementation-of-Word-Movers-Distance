#!/usr/bin/env python

"""
    sinkhorn_wmd/main.py
    
    !! Note to program performers  !! 
    
    There are minor differences between 
        this implementation and the one that was previously released as an program workflow.
        
        1) We just run for a fixed number of iterations 
            here, for ease of reproducibility
        2) The `reg` parameter in the previous implementation 
            is equivilant to `1 / lamb` here to match pseudocode
    
    The computation of each iteration is the same, though we're using standard
    elementwise multiplication and matrix multiplication instead of the custom
    numba kernels from the previous version, for ease of explanation.  If you find
    edge cases where this is producing different results from the program workflow, 
    please let me know.
    
    ~ Ben Johnson (bkj.322@gmail.com)
"""

import os
import sys
import argparse

try:
    import cunumeric as np
    import sparse
    # from scipy import sparse
except ImportError:
    import numpy as np
    from scipy import sparse

try:
    from legate.timing import time
except ImportError:
    from time import perf_counter_ns

    def time():
        return perf_counter_ns() / 1000.0

from scipy import sparse as scipy_sparse
from scipy.spatial.distance import cdist

# --
# Sinkhorn

#@profile
def sinkhorn_wmd(r, c, vecs, lamb, max_iter):
    """
        r (np.array):          query vector (sparse, but represented as dense)
        c (sparse.csr_matrix): data vectors, in CSR format.  shape is `(dim, num_observations)`
        vecs (np.array):       embedding vectors, from which we'll compute a distance matrix
        lamb (float):          regularization parameter -- note this is (1 / reg) from previous original implementation
        max_iter (int):        maximum number of iterations
        
        Inline comments reference pseudocode from Alg. 1 in paper
            https://arxiv.org/pdf/1306.0895.pdf
    """

    sel = r.squeeze() > 0
    r = r[sel].reshape((-1, 1)).astype(np.float64)

    # TODO (rohany): We probably need a distributed implementation of cdist.
    M = cdist(vecs[sel], vecs).astype(np.float64)
    
    a_dim  = r.shape[0]
    b_nobs = c.shape[1]
    x      = np.ones((a_dim, b_nobs)) / a_dim 
    
    K = np.exp(M * lamb)
    p = (1 / r) * K
    KT = K.T
    KM = (K * M)

    it = 0
    while it < max_iter:
        u = 1.0 / x
        v = c.multiply(1 / (KT @ u))

        # x = p @ v
        x = v.__rmatmul__(p)
        it += 1
    
    u = 1.0 / x
    v = c.multiply(1 / (KT @ u))
    # return (u * ((KM) @ v)).sum(axis=0)
    return (u * (v.__rmatmul__(KM))).sum(axis=0)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=-1)
    parser.add_argument('--max_iter', type=int, default=15)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.inpath == 'data/cache'
    assert args.n_docs == 5000
    assert args.query_idx == 100
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Load input data.
    vecs = np.load(args.inpath + '-vecs.npy')
    mat = scipy_sparse.load_npz(args.inpath + '-mat.npz').tocsr()
    
    
    # Maybe subset docs.
    if args.n_docs:
        mat  = mat[:,:args.n_docs]
    
    # Get query vector.
    r = np.asarray(mat[:,args.query_idx].todense()).squeeze()
   
    # Convert the loaded scipy sparse matrix into a legate sparse matrix.
    mat = sparse.csr_array(mat)

    t = time()
    scores = sinkhorn_wmd(r, mat, vecs, lamb=args.lamb, max_iter=args.max_iter)
    elapsed = time() - t
    print('elapsed=%f ms.' % (elapsed / 1000.0), file=sys.stderr)
    
    # Write output.
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/scores', scores, fmt='%.8f')
    open('results/elapsed', 'w').write(str(elapsed))
