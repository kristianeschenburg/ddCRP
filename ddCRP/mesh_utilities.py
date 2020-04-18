import networkx as nx
import numpy as np
from scipy import sparse

"""
Methods / classes to aid in mesh (pre)-processing.
"""


def unweightedadjacency(F, to_dict=True):
    """
    Compute adjacency structure of a mesh.
    Generates either a sparse matrix or an adjacency list.

    Parameters:
    - - - - -
    F: int, array
        mesh triangles

    Returns:
    - - - -
    A: scipy.sparse.csr_matrix / dict
        adjacency structure of mesh
    """

    n = F.max()+1

    rows = np.concatenate([F[:, 0], F[:, 0],
                           F[:, 1], F[:, 1],
                           F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2],
                           F[:, 0], F[:, 2],
                           F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [_, idx] = np.unique(combos, axis=0, return_index=True)
    print('# unique vertices: %i' % (len(idx)))
    A = sparse.csr_matrix(
        (np.ones(len(idx)), (combos[idx, 0], combos[idx, 1])), shape=(n, n))
    
    if to_dict:
        A_dict = {k: [] for k in np.arange(A.shape[0])}
        
        for k in A_dict.keys():
            A_dict[k] = list(sparse.find(A[k])[1])
    
        A = A_dict

    return A

def find_boundaries(index_map, adj_list):

    """
    Method to identify vertices that exist at the boundary of two regions.

    Parameters:
    - - - - -
    index_map: dictionary
        mapping of region names / classes to indices assigned to that class

    adj_list : dictionary
        adjacency list for surface mesh on which label map lives

    Returns:
    - - - -
    boundaries: dict
        mapping of each region / class to a set
        of indices that exist at the boundary of that class.
    """

    boundaries = {k: None for k in index_map.keys()}

    for region, inds in index_map.items():

        binds = []

        for tidx in inds:

            neighbors = set(adj_list[tidx])
            outer = neighbors.difference(set(inds))
            if len(outer) > 0:
                binds.append(tidx)
        
        boundaries[region] = binds
    
    return boundaries
