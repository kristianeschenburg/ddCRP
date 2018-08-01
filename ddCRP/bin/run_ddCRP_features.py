import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import argparse
import sys
from niio import loaded, write
import nibabel as nb
import numpy as np

import scipy.io as sio

from ..ddCRP.ddCRP import ddCRP
from ..ddCRP.subgraphs import ClusterSpanningTrees

# codedir = '/mnt/parcellator/parcellation/GitHub/'
codedir = '/Users/kristianeschenburg/Documents/Code/'
sys.path.append(''.join([codedir,
                        'surface_utilities/surface_utilities/']))
sys.path.append(''.join([codedir,
                        'Parcellating-connectivity/python/model_similarity/']))

import adjacency as adj
import WardClustering


parser = argparse.ArgumentParser()
parser.add_argument('-surf', '--surfacefile', help='Surface to compute '
                    'adjacency on.', required=True, type=str)
parser.add_argument('-lab', '--labelfile', help='Label map to get '
                    'indices from.', required=True, type=str)
parser.add_argument('-data', '--datafile', help='Data to cluster.',
                    required=True, type=str)

parser.add_argument('-a', '--alpha', help='Dirichlet process concentration '
                    'parameter.', required=True, type=float)
parser.add_argument('-m', '--mu', help='Prior on mean expected value.',
                    required=True, type=float)
parser.add_argument('-k', '--kappa', help='Prior on mean variance.',
                    required=True, type=float)
parser.add_argument('-n', '--nu', help='Prior on variance expected value.',
                    required=True, type=float)
parser.add_argument('-s', '--sigma', help='Prior on variance variance.',
                    required=True, type=float)

parser.add_argument('-w', '--ward', help='Booleant to run Ward clustering '
                    'before ddCRP.', required=True, default=False, type=bool)
parser.add_argument('-mcmc', '--monte', help='Number of MCMC iterations '
                    'to run.', required=False, type=int, default=75)

parser.add_argument('-out', '--outbase', help='Output file base name.',
                    required=True, type=str)

# Parse supplied arguments
args = parser.parse_args()

# Prepare region indices
print('Preparing region indices.')
regions = ['L_inferiorparietal', 'L_supramarginal']
label = nb.load(args.labelfile)
cdata = label.darrays[0].data
lt = label.get_labeltable().get_labels_as_dict()

reg2val = dict(zip(map(str, lt.values()), lt.keys()))
regions = ['L_inferiorparietal', 'L_supramarginal']

inds = []
for r in regions:
    inds.append(np.where(cdata == reg2val[r])[0])
inds = np.concatenate(inds)
inds.sort()
print('{:} indices in map.'.format(len(inds)))

print('Preparing adjacency list.')
S = adj.SurfaceAdjacency(args.surfacefile)
S.generate()
filtered = S.filtration(filter_indices=inds)

# remap indices to sorted order
inds2sort = dict(zip(inds, np.arange(len(inds))))
adj_list = {inds2sort[k]: [inds2sort[v] for v in filtered[k]]
            for k in filtered.keys()}

print('Preparing features.')
features = loaded.load(args.datafile)
[n, p] = features.shape
if p > n:
    features = features.T

features = features[inds, :]
print('Features shape: {:}'.format(features.shape))

mu_f = features.mean(0)
std_f = features.std(0)

# find where features are all zero
zero_std = std_f == 0

# normalize columns and exclude all-zero features
features_norm = (features - mu_f[None, :]) / std_f[None, :]
features_norm = features_norm[:, ~zero_std]

alpha = args.alpha
mu = args.mu
kappa = args.kappa
nu = args.nu
sigma = args.sigma
ward = args.ward

if ward:
    print('Initializing linkage matrix with Ward Clustering.')
    f_corr = np.corrcoef(features_norm)
    Z = WardClustering.ClusterTree(f_corr, adj_list)
    z = WardClustering.Cluster(Z, n=7)
    c = ClusterSpanningTrees(adj_list, z).fit()
    c = c.astype(np.int32)
else:
    c = []

print('Fitting ddcrp.')
print('Initial linkage: {:}'.format(np.any(c)))
crp = ddCRP(alpha, mu, kappa, nu, sigma, mcmc_passes=args.monte,
            stats_interval=200)
crp.fit(features_norm, adj_list, init_c=c)

print('Saving model results.')

keys = ['alpha', 'mu', 'kappa', 'nu', 'sigma', 'ward']
param_map = dict(zip(keys, [alpha, mu, kappa, nu, sigma, ward]))

param_tuple = [(k, param_map[k]) for k in keys]
extension = ''.join("%s." % '.'.join(map(str, x)) for x in param_tuple)
outbase = '.'.join([args.outbase, extension])

print('Saving statistics.')
sio.savemat(''.join([outbase, 'statistics.mat']), mdict=crp.stats)

print('Saving MAP label.')
full_mapz = np.zeros((len(cdata),))
print('FullMap shape: {:}'.format(full_mapz.shape))
full_mapz[inds] = crp.map_z
write.save(full_mapz, ''.join([outbase, 'map_z.func.gii']), 'CortexLeft')

print('Saving figures.')
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(16, 5))

ax1.plot(np.arange(1, len(crp.stats['K'])+1), crp.stats['K'])
ax1.set_title('Cluster Count', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Clusters', fontsize=15)
ax2.plot(np.arange(1, len(crp.stats['lp'])+1), crp.stats['lp'])
ax2.set_title('Log Probability', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('lp', fontsize=15)
ax3.plot(np.arange(1, len(crp.stats['max_lp'])+1), crp.stats['max_lp'])
ax3.set_title('Max Log-Probability', fontsize=15)
ax3.set_xlabel('Iteration', fontsize=15)
ax3.set_ylabel('lp', fontsize=15)
plt.tight_layout()
plt.savefig(''.join([outbase, 'jpg']))
