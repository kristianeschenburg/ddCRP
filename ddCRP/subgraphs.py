import numpy as np
from scipy.sparse import csgraph, csc_matrix
import random


def ConnectedComponents(G):
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

    [K, components] = csgraph.connected_components(G, directed=False,
                                                   connection='weak')

    sorted_i = np.argsort(components)
    sorted_z = components[sorted_i]

    parcels = np.split(sorted_i, np.flatnonzero(np.diff(sorted_z))+1)
    parcels = dict(zip(list(set(sorted_z)), map(list, parcels)))

    return [K, components, parcels]


class ClusterSpanningTrees(object):
    """
    Compute cluster-specific minimum spanning trees.

    Parameters:
    - - - - -
        adj_list : dictionary
                    adjacency list of samples
        z : array
            initial clustering
    """

    def __init__(self, adj_list, z):
        self.adj_list = adj_list
        self.z = z.astype(np.int32)

    def fit(self):
        """
        Compute connected components of graph.

        Returns:
        - - - -
            c : array
                mininium spanning trees of clustering, where index
                c[i] = parent of node i
        """

        self._filter_adjacency()
        c = self._construct_sparse()

        return c

    def _filter_adjacency(self):
        """
        Filter an adjacency list so that there are no cross-cluster edges.

        Parameters:
        - - - - -
            adj_list : dictionary
                        input adjacency list
            z : array
                input clustering
        """

        adj_list = self.adj_list.copy()

        for k, v in adj_list.items():
            adj_list[k] = list(np.asarray(v)[self.z[v] == self.z[k]])
            adj_list[k] = list(np.random.permutation(adj_list[k]))

        self.adj_list = adj_list

    def _construct_sparse(self):
        """
        Construct sparse matrix from adjacency list.

        Returns:
        - - - -
            c : array
                minimum spanning trees of clustering
        """
        nvox = len(self.adj_list.keys())
        neighbor_count = [len(self.adj_list[k]) for k in self.adj_list.keys()]
        node_list = np.zeros(sum(neighbor_count))
        next_edge = 0

        # repeat i as many times as it has neighbors
        for i in np.arange(nvox):
            # if vertex has more than one neighbor
            if neighbor_count[i] > 0:
                node_list[next_edge:(next_edge+neighbor_count[i])] = i
                next_edge += neighbor_count[i]

        node_list = list(map(int, node_list))

        G = csc_matrix((
            np.ones(len(node_list)),
            (node_list, np.hstack(self.adj_list.values()))),
            shape=(nvox, nvox))

        # Construct spanning tree in each cluster
        minT = csgraph.minimum_spanning_tree(G)
        c = np.zeros(len(self.adj_list))
        for clust in np.unique(self.z):
            clust_vox = np.flatnonzero(self.z == clust)
            rand_root = clust_vox[random.randint(1, len(clust_vox)-1)]
            _, parents = csgraph.breadth_first_order(minT, rand_root,
                                                     directed=False)
            c[clust_vox] = parents[clust_vox]

        # Roots have parent value of -9999, set them to be their own parent
        roots = np.flatnonzero(c == -9999)
        c[roots] = roots

        return c
