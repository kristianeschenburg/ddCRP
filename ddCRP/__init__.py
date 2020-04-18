from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import (ddCRP, mesh_utilities, PriorBase, Priors, sampling, statistics, subgraphs, synthetic, utilities, ward_clustering)