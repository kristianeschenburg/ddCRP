import numpy as np
import random

class Gibbs(object):

	"""
	Simple class to perform Gibbs sampling.
	"""

	def __init__(self):

		pass

	def fit(self,lp):

		max_lp = lp.max()
		normLogP = lp - (max_lp + np.log(np.exp(lp-max_lp).sum()))
		p = np.exp(normLogP)
		p[np.isnan(p)] = 0
		p[np.isinf(p)] = 0

		cumP = np.cumsum(p)
		i = np.where(cumP>random.random())[0][0]

		return i