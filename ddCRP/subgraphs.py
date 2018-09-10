import numpy as np
from scipy import sparse
import random

"""
As based on original code by C. Baldassano (https://github.com/cbaldassano/Parcellating-connectivity/blob/release/python/WardClustering.py)
"""


def connected_components(G):
    """
    Compute connected components of graph.

    Parameters:
    - - - - -
    G : array
        sparse graph of current clustering

Returns:
- - - -
    K : int
        number of components
    z : array
        cluster assignment of each sample
    parcels : dictionary
        mapping between cluster ID and sample indices
    """

    [K, components] = sparse.csgraph.connected_components(G, directed=False,
                                                   connection='weak')

    sorted_i = np.argsort(components)
    sorted_z = components[sorted_i]

    parcels = np.split(sorted_i, np.flatnonzero(np.diff(sorted_z))+1)
    parcels = dict(zip(list(set(sorted_z)), map(list, parcels)))

    return [K, components, parcels]


def remap_components(G, label):
    """
    Check to make sure that the label map has the same number of labels
    as there are connected components.  If not, remap. the connected components.

    Parameters:
    - - - -- -
    G: sparse adjacency matrix
        adjacency structure after filtering cross-label edges

    Returns:
    - - - -
    label: array
        original or remapped label array
    """

    L = len(np.unique(label))
    [K, r] = sparse.csgraph.connected_components(G)

    if K != L:
        label = r

    return label


def sparse_linkage(adj_list, nvox, init_c=None):
        """
        Compute source-to-target linkages and sparse neighborhood matrix.
        Parameters:
        - - - - -
        adj_list: dictionary
            adjacency list of samples
        nvox: int
            number of samples
        init_c: array
            initial source-to-taret linkages
        """

        if not np.any(init_c):
            c = np.zeros((nvox,))
            for i in np.arange(nvox):
                neighbors = adj_list[i] + [i]
                c[i] = neighbors[np.random.randint(low=0, high=len(neighbors))]
        else:
            c = init_c

        c = c.astype(np.int32)

        # initialize sparse linkage matrix
        G = sparse.csc_matrix(
            (np.ones((nvox, )), (np.arange(nvox), c)), shape=(nvox, nvox))

        G = G.tolil()

        return [c, G]


class ClusterSpanningTrees(object):
    """
    Compute cluster-specific minimum spanning trees.

    """

    def __init__(self):
        pass

    def fit(self, adj_list, z):
        """
        Compute connected components of graph.

        Parameters:
        - - - - -
        adj_list : dictionary
                    adjacency list of samples
        z : array
            initial clustering

        Returns:
        - - - -
        c : array
            mininium spanning trees of clustering, where index
            c[i] = parent of node i
        """

        filtered = self.filter_by_label(adj_list, z)
        c = self.construct_sparse(filtered, z)

        return c

    @staticmethod
    def filter_by_label(adjacency, label):
        """
        Filter an adjacency list so that there are no cross-cluster edges.

        Parameters:
        - - - - -
        adj_list : dictionary
            input adjacency list
        z : array
            input clustering
        """

        adj_list = adjacency.copy()

        for k, v in adj_list.items():
            adj_list[k] = list(np.asarray(v)[label[v] == label[k]])
            adj_list[k] = list(np.random.permutation(adj_list[k]))

        return adj_list

    @staticmethod
    def construct_sparse(adjacency, label):
        """
        Construct sparse matrix from adjacency list.

        Returns:
        - - - -
        c : int array
            minimum spanning trees of clustering
        """

        # Construct sparse adjacency matrix
        nvox = len(adjacency.keys())
        neighbor_count = [len(adjacency[k])
                          for k in adjacency.keys()]
        node_list = np.zeros(sum(neighbor_count))
        next_edge = 0

        # repeat i as many times as it has neighbors
        for i in np.arange(nvox):
            # if vertex has more than one neighbor
            if neighbor_count[i] > 0:
                node_list[next_edge:(next_edge+neighbor_count[i])] = i
                next_edge += neighbor_count[i]

        node_list = list(map(int, node_list))

        sources = np.concatenate([node_list, np.hstack(adjacency.values())])
        targets = np.concatenate([np.hstack(adjacency.values()), node_list])

        G = sparse.csc_matrix((
            np.ones(len(sources)),
            (sources, targets)),
            shape=(nvox, nvox))
        
        # Check to make sure # labels = # components
        # otherwise, remap labels
        label = remap_components(G, label)

        # Construct spanning tree in each cluster
        minT = sparse.csgraph.minimum_spanning_tree(G)
        c = np.zeros(len(adjacency))
        for clust in np.unique(label):
            clust_vox = np.flatnonzero(label == clust)
            rand_root = np.random.choice(clust_vox)
            _, parents = sparse.csgraph.breadth_first_order(minT, rand_root,
                                                     directed=False)
            c[clust_vox] = parents[clust_vox]

        # Roots have parent value of -9999, set them to be their own parent
        roots = np.flatnonzero(c == -9999)
        c[roots] = roots

        return c

