import numpy as np
import random

"""
As based on original code by C. Baldassano (https://github.com/cbaldassano/Parcellating-connectivity/blob/release/python/ddCRP.py)
"""


class Gibbs(object):
    """
    Simple class to perform Gibbs sampling.
    """

    def __init__(self):

        pass

    def sample(self, lp):


        max_lp = lp.max()
        normLogP = lp - (max_lp + np.log(np.exp(lp-max_lp).sum()))
        p = np.exp(normLogP)
        p[np.isnan(p)] = 0
        p[np.isinf(p)] = 0
        cumP = np.cumsum(p)
        i = np.where(cumP > random.random())[0][0]

        return i
