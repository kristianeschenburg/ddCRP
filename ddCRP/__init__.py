from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

<<<<<<< HEAD
from . import (ddCRP, mesh_utilities, PriorBase, Priors, sampling, statistics, subgraphs, synthetic, utilities, ward_clustering)
=======

from . import (PriorBase, Priors, statistics, subgraphs, synthetic, ward_clustering, adjacency)
>>>>>>> 2aa12a279188257b9c22265a04094e5d9af7cb0e
