import numpy as np
import time

def UpdateStats(stats, t0, curr_lp, max_lp, K, z, c, steps, gt_z, map_z, verbose):

	"""
	Update diagnostic statistics.

	Parameters:
	- - - - -
		t0 : initial start time
		curr_lp : current log-probability of map
		max_lp : max log-probability
		K : number of clusters
		z : current map
		c : current parent links
		steps : total number of steps taken
		gt_z : ground truth map
		map_z : maximum a-posterior map
		verbose : flag to print status updates
	"""

    
	stats['lp'].append(curr_lp)
	stats['max_lp'].append(max_lp)
	stats['K'].append(K)
	stats['z'] = np.row_stack([stats['z'],z])
	stats['c'] = np.row_stack([stats['c'],c])
	curr_time = time.clock() - t0
	stats['times'].append(curr_time)
	if verbose:
	    print('Step: ' + str(steps) + ' Time: ' + str(curr_time) + 
	            ' LP: ' + str(curr_lp) + ' K: ' + str(K) + ' MaxLP: ' + str(max_lp))

	if np.any(gt_z):
		stats['NMI'].append(NMI(gt_z, map_z))

	return stats

def NMI(z1, z2):

	"""
	Compute normalized mutual information between two maps.two

	Parameters:
	- - - - -
		z1, z2 : maps to compare
	"""

	N = len(z1)
	assert N == len(z2)

	p1 = np.bincount(z1)/N
	p1[p1 == 0] = 1
	H1 = (-p1*np.log(p1)).sum()

	p2 = np.bincount(z2)/N
	p2[p2 == 0] = 1
	H2 = (-p2*np.log(p2)).sum()

	joint = np.histogram2d(z1,z2,[range(0,z1.max()+2), range(0,z2.max()+2)],
																	normed=True)
	joint_p = joint[0]
	pdiv = joint_p/np.outer(p1,p2)
	pdiv[joint_p == 0] = 1
	MI = (joint_p*np.log(pdiv)).sum()

	if MI == 0:
		NMI = 0
	else:
		NMI = MI/np.sqrt(H1*H2)

	return NMI