import numpy as np
import time
from surface_adjacency import adjacency

def UpdateStats(stats, t0, curr_lp, max_lp, K, z, c, steps,
	gt_z, map_z, deltaC, boundC, verbose=True):

	"""
	Update diagnostic statistics.
	Parameters:
	- - - - -
	t0 : float
		initial start time
	curr_lp : float
		current log-probability of map
	max_lp : float
		max log-probability
	K : int
		number of clusters
	z : array
		current map
	c : array
		current parent links
	steps : int
		total number of steps taken
	gt_z : array
		ground truth map
	map_z : array
		maximum a-posterior map
	deltaC : int
		number of vertices that changed label at current iteration
	boundC : int
		number of boundary vertices at current iteration
	verbose : bool
		flag to print status updates
	"""
	stats['lp'].append(curr_lp)
	stats['max_lp'].append(max_lp)
	stats['K'].append(K)
	stats['z'] = np.row_stack([stats['z'], z])
	stats['c'] = np.row_stack([stats['c'], c])
	curr_time = time.clock() - t0
	stats['times'].append(curr_time)
	stats['deltaC'].append(deltaC)
	stats['boundC'].append(boundC)

	if verbose:
		print(
			'Step: '+str(steps) + ' Time: ' + str(curr_time) +
			' LP: ' + str(curr_lp) + ' K: ' + str(K) + ' MaxLP: ' +
			str(max_lp))

	if np.any(gt_z):
		stats['NMI'].append(NMI(gt_z, map_z))

	return stats


def boundaries(label, adj_list, normed=True):
	"""
	Compute number of boundary vertices in label map.

	Parameters:
	- - - - -
	label : array
		cortical map
	adjacency : dictionary
		adjacency list
	Returns:
	- - - -
	boundC : int
		number of vertics that exist at parcel boundaries
	"""

	boundC = adjacency.BoundaryMap(label, adj_list).find_boundaries()
	boundC = boundC.sum()

	if normed:
		boundC = boundC / len(label)

	return boundC


def delta_C(parcels_old, parcels_new, normed=False):
	"""
	Compute the number of vertices that change connected component from
	old parcellation to new parcellation.
	Parameters:
	- - - - -
	parcels_old : dictionary
			old connected component sample assignments
	parcels_new : dictionary
			new connected component sample assignments
	Returns:
	- - - -
	deltaC : int
		number of vertices that changed label
	"""

	new = set(map(len, parcels_new.values()))
	old = set(map(len, parcels_old.values()))

	deltaC = np.int32(list(new.difference(old))).sum()

	if normed:
		deltaC = deltaC / np.sum(list(new))

	return deltaC


def Normalize(D):
	"""
	Method to normalize feature matrix.
	Parameters:
	- - - - -
	D : array
		input feature matrix
	Returns:
	- - - -
	D_norm : array
		normalized feature matrix
	"""

	mu = D.mean(0)
	stdev = D.std(0)

	zs = (stdev != 0)

	D_norm = (D[:, zs] - mu[zs][None, :]) / stdev[zs][None, :]

	return D_norm


def NMI(z1, z2):
	"""
	Compute normalized mutual information between two maps.
	Parameters:
	- - - - -
	z1, z2 : array
		maps to compare
	"""

	N = len(z1)
	assert N == len(z2)

	p1 = np.bincount(z1)/N
	p1[p1 == 0] = 1
	H1 = (-p1*np.log(p1)).sum()

	p2 = np.bincount(z2)/N
	p2[p2 == 0] = 1
	H2 = (-p2*np.log(p2)).sum()

	joint = np.histogram2d(
		z1, z2, [np.arange(0, z1.max()+2), np.arange(0, z2.max()+2)],
		normed=True)

	joint_p = joint[0]
	pdiv = joint_p/np.outer(p1, p2)
	pdiv[joint_p == 0] = 1
	MI = (joint_p*np.log(pdiv)).sum()

	if MI == 0:
		NMI = 0
	else:
		NMI = MI/np.sqrt(H1*H2)

	return NMI
