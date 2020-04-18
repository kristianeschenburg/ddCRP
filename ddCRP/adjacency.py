class SurfaceAdjacency(object):
    """
    Class to generate an adjancey list  of a surface mesh representation
    of the brain.

    Initialize SurfaceAdjacency object.

    Parameters:
    - - - - -
    vertices : array
        vertex coordinates
    faces : list
        list of faces in surface
    """

    def __init__(self, vertices, faces):

        self.vertices = vertices
        self.faces = faces

    def generate(self, indices=None):
        """
        Method to create surface adjacency list.
        """

        # Get faces attribute
        faces_ = []
        faces = self.faces.tolist()
        accepted = np.zeros((self.vertices.shape[0]))

        # get indices of interest
        if not np.any(indices):
            indices = list(np.unique(np.concatenate(faces)))
        indices = np.sort(indices)

        print('Num indices: {:}'.format(len(indices)))
        print('Num vertices: {:}'.format(self.vertices.shape[0]))

        # create array of whether indices are included
        # cancels out search time
        accepted[indices] = 1
        accepted = accepted.astype(bool)

        # Initialize adjacency list
        adj = {k: [] for k in indices}

        # loop over faces in mesh
        for face in faces:
            for j, vertex in enumerate(face):
                idx = (np.asarray(face) != vertex)
                if accepted[vertex]:
                    nbs = [n for n in np.asarray(face)[idx] if accepted[n]]
                    adj[face[j]].append(nbs)

            if len(indices) != self.vertices.shape[0]:
                inter = np.intersect1d(indices, face)
                if len(inter) == 3:
                    faces_.append(face)

        for k in adj.keys():
            if adj[k]:
                adj[k] = list(set(np.concatenate(adj[k])))

        # Set adjacency list field
        self.adj = adj
        self.faces_ = np.asarray(faces_)
        self.vertices_ = self.vertices[indices, :]

    @staticmethod
    def filtration(adj, filter_indices, toArray=False, remap=False):
        """
        Generate a local adjacency list, constrained to a subset of vertices on
        the surface.  For each vertex in 'vertices', retain neighbors
        only if they also exist in 'vertices'.

        Parameters:
        - - - - -
        adj : dictionary
            adjacency list to filter
        fitler_indices : array
            indices to include in sub-graph.  If none, returns original graph.
        to_array : bool
            return adjacency matrix of filter_indices
        remap : bool
            remap indices to 0-len(filter_indices)
        Returns:
        - - - -
        G : array / dictionary
            down-sampled adjacency list / matrix
        """

        accepted = np.zeros((len(adj.keys()),))
        accepted[filter_indices] = True

        filter_indices = np.sort(filter_indices)

        G = {}.fromkeys(filter_indices)

        for v in filter_indices:
            neighbors = adj[v]
            neighbors = [n for n in neighbors if n in filter_indices]
            G[v] = list(set(adj[v]).intersection(set(filter_indices)))

        ind2sort = dict(zip(
            filter_indices,
            np.arange(len(filter_indices))))

        if remap:
            remapped = {
                ind2sort[fi]: [ind2sort[nb] for nb in G[fi]]
                for fi in filter_indices}

            G = remapped

        if toArray:
            G = nx.from_dict_of_lists(G)
            nodes = G.nodes()
            nodes = np.argsort(nodes)
            G = nx.to_numpy_array(G)
            G = G[nodes, :][:, nodes]

        return G

class BoundaryMap(object):

    """
    Class to find vertices that exist on the boundary of two regions.

    Parameters:
    - - - - -
        label : label map from which to compute borders
        adj_list : adjacency list for mesh
    """

    def __init__(self, label, adj_list):

        self.label = label
        self.adj_list = adj_list

    def find_boundaries(self):

        """
        Method to identify vertices that exist at the boundary of two regions.
        """

        adj_list = self.adj_list
        label = self.label

        boundaries = np.zeros((label.shape))

        for i, k in enumerate(adj_list.keys()):

            neighbors = adj_list[k]
            nLabels = list(set(label[neighbors]) - set([0]))

            if len(set(nLabels)) > 1:
                boundaries[i] = 1

        self.boundaries = boundaries