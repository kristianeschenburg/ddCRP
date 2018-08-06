from niio import loaded, write
from ddCRP import ddCRP
from surface_utilities import label_utilties, adjacency

import numpy as np
import nibabel as nib

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-surf', 'surface', help='Input surface file.',
	required=True, type=str)
parser.add_argument('-lab', '--label', help='Input label file.',
	required=True, type=str)
parser.add_argument('-data', '--features', help='Input features to cluster.',
	required=True, type=str)
parser.add_argument('-a', '--alpha', help='Concentration parameter.',
	required=True, type=str)
parser.add_argument('-m', '--mu', help='Prior on mean.',
	required=True, type=float)
parser.add_argument('-k', '--kappa', help='Prior on mean.',
	required=True, type=float)
parser.add_argument('-n', '--nu', help='Prior on variance.',
	required=True, type=float)
parser.add_argument('-s', '--sigma', help='Prior on variance.',
	required=True, type=float)
parser.add_argument('-w', '--ward', help='Boolean to apply Ward Clustering.',
	required=False, type=bool, default=True)
parser.add_argument('-mcmc', '--montecarlo', help='Number of MCMC passes.',
	required=False, type=int, default=100)

args = parser.parse_args()

# load label, get sorted indices
lab_obj = nib.load(args.label)
cdata = lab_obj.darrays[0].data
regions = ['L_inferiorparietal', 'L_supramarginal']
indices = np.sort(label_utilties.region_indices(lab_obj, regions))

# load features, filter by indices
data = loaded.load(args.features)
n, p = data.shape
if p > n:
	data = data.T
features = data[indices, :]

# load surface, filter and remap adjacency list
S = adjacency.SurfaceAdjacency(args.surface)
S.generate()
adj_list = S.filtration(filter_indices=indices, remap=True)

# get hyperparameters
alpha = args.alpha
mu = args.mu
kappa = args.kappa
nu = args.nu
sigma = args.sigma

mcmc = args.montecarlo

# initialize and fit ddCRP model
crp = ddCRP.ddCRP(alpha=alpha, mu_0=mu, kappa_0=kappa, nu_0=nu, sigma_0=sigma,
	ward=args.ward, mcmc_passes=mcmc)

crp.fit(features=features, adj_list=adj_list)
