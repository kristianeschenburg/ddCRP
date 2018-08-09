from niio import loaded, write
from ddCRP import ddCRP
from surface_utilities import label_utilties, adjacency

import numpy as np
import nibabel as nib
import scipy.io as sio

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


parser = argparse.ArgumentParser()

parser.add_argument(
    '-surf', 'surface', help='Input surface file.', required=True, type=str)
parser.add_argument(
    '-lab', '--label', help='Input label file.', required=True, type=str)
parser.add_argument(
    '-data', '--features', help='Input features to cluster.', required=True,
    type=, nargs='+')
parser.add_argument(
    '-a', '--alpha', help='Concentration parameter.', required=True, type=str)
parser.add_argument(
    '-m', '--mu', help='Prior on mean.', required=True, type=float)
parser.add_argument(
    '-k', '--kappa', help='Prior on mean.', required=True, type=float)
parser.add_argument(
    '-n', '--nu', help='Prior on variance.', required=True, type=float)
parser.add_argument(
    '-s', '--sigma', help='Prior on variance.', required=True, type=float)
parser.add_argument(
    '-w', '--ward', help='Boolean to apply Ward Clustering.', required=False,
    type=bool, default=True)
parser.add_argument(
    '-mcmc', '--montecarlo', help='Number of MCMC passes.', required=False,
    type=int, default=100)
parser.add_argument(
    '-gt', '--ground', help='Ground truth map.', required=False, type=str,
    default=None)

args = parser.parse_args()

# load label, get sorted indices
lab_obj = nib.load(args.label)
cdata = lab_obj.darrays[0].data
regions = ['L_inferiorparietal', 'L_supramarginal']
indices = np.sort(label_utilties.region_indices(lab_obj, regions))

# load features, filter by indices
features = []
for feat_type in args.features:
    temp = loaded.load(feat_type)
    n, p = temp.shape
    if p > n:
        temp = temp.T
    temp = temp[indices, :]
features.append(temp)
features = np.column_stack(features)

if args.gt:
    gt = loaded.load(args.gt)
    gt = gt[indices]
else:
    gt = args.gt

# load surface, filter and remap adjacency list
surface = nib.load(args.surface)
vertices = surface.darrays[0].data
faces = surface.darrays[1].data

S = adjacency.SurfaceAdjacency(vertices, faces)
S.generate()
adj_list = S.filtration(
    filter_indices=indices, remap=True)

# get hyperparameters
alpha = args.alpha
mu = args.mu
kappa = args.kappa
nu = args.nu
sigma = args.sigma

mcmc = args.montecarlo
ward = args.ward

# initialize and fit ddCRP model
crp = ddCRP.ddCRP(alpha=alpha, mu_0=mu, kappa_0=kappa, nu_0=nu, sigma_0=sigma,
                  ward=ward, mcmc_passes=mcmc)

crp.fit(features=features, adj_list=adj_list, gt_z=gt)


print('Saving model results.')

keys = ['alpha', 'mu', 'kappa', 'nu', 'sigma', 'ward']
param_map = dict(zip(keys, [alpha, mu, kappa, nu, sigma, ward]))

param_tuple = [(k, param_map[k]) for k in keys]
extension = ''.join("%s." % '.'.join(map(str, x)) for x in param_tuple)
outbase = '.'.join([args.outbase, extension])


print('Saving statistics.')
sio.savemat(''.join([outbase, 'statistics.mat']), mdict=crp.stats_)


print('Saving MAP label.')
full_mapz = np.zeros((len(cdata),))
print('FullMap shape: {:}'.format(full_mapz.shape))
full_mapz[indices] = crp.map_z_
write.save(full_mapz, ''.join([outbase, 'map_z.func.gii']), 'CortexLeft')


print('Saving figures.')
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(1, 3, figsize=(16, 5))

# plot cluster counts
ax1.plot(np.arange(1, len(crp.stats_['K'])+1), crp.stats_['K'])
ax1.set_title('Cluster Count', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Clusters', fontsize=15)

# plot log-probability
ax2.plot(np.arange(1, len(crp.stats_['lp'])+1), crp.stats_['lp'])
ax2.set_title('Log Probability', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('lp', fontsize=15)

# plot max log-probability
ax3.plot(np.arange(1, len(crp.stats_['max_lp'])+1), crp.stats_['max_lp'])
ax3.set_title('Max Log-Probability', fontsize=15)
ax3.set_xlabel('Iteration', fontsize=15)
ax3.set_ylabel('lp', fontsize=15)

# plot fraction of of boundary samples
ax4.plot(np.arange(1, len(crp.stats_['boundC'][1:]), crp.stats_['boundC'][1:]))
ax4.set_title('Boundary Vertex Fraction', fontsize=15)
ax4.set_xlabel('Iteration', fontsize=15)
ax4.set_ylabel('Boundary Fraction', fontsize=15)

# plot fraction of samples that change label
ax5.plot(np.arange(1, len(crp.stats_['deltaC'][1:]), crp.stats_['deltaC'][1:]))
ax5.set_title('Label Change Fraction', fontsize=15)
ax5.set_xlabel('Iteration', fontsize=15)
ax5.set_ylabel('Label Changes', fontsize=15)

plt.tight_layout()
plt.savefig(''.join([outbase, 'jpg']))
