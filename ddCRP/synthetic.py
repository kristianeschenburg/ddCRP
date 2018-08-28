import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import invgamma, norm, multivariate_normal

"""
As based on oringal code by C. Baldassano (https://github.com/cbaldassano/Parcellating-connectivity/blob/release/python/LearnSynth.py)
"""

SYNTH_TYPES = ['square', 'ell', 'stripes', 'face']


class SampleSynthetic(object):
    """
    Sample synthetic labels and features for these data.

    See https://github.com/cbaldassano/Parcellating-connectivity/
    for more details.

    Parameters:
    - - - - -
    kind : string
        shape of true classes
    d : int
        feature dimensions
    mu_0 : float
        prior mean of first moment
    kappa_0 : float
        prior variance of first moment
    nu_0 : float
        prior mean of second moment
    sigma_0 : float
        prior variance of second moment
    """

    def __init__(self, kind, d, mu0, kappa0, nu0, sigma0):

        assert kind in SYNTH_TYPES

        self.kind = kind
        self.d = d
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.sigma0 = sigma0

    def fit(self):

        """
        Fit prior models and generate synthetic data.
        """

        self.synthetic_labels()
        [pcs, prms, pf, f] = self.synthetic_features(self.z_)

        self.parcels_ = pcs
        self.params_ = prms
        self.parcel_features_ = pf
        self.features_ = f

    def synthetic_labels(self):
        """
        Generate synthetic label array.

        Parameters:
        - - - - -
            type : string
                    type of synthetic data to generate
        """
        sqrtN = 18

        coords = np.zeros((sqrtN**2, 2))
        adj_list = {}.fromkeys(np.arange(sqrtN**2))
        for r in range(0, sqrtN):
            for c in range(0, sqrtN):
                currVox = c + r*sqrtN
                coords[currVox, :] = [r, c]
                curr_adj = []
                if r > 0:
                    curr_adj.append(c + (r-1)*sqrtN)
                if r < (sqrtN-1):
                    curr_adj.append(c + (r+1)*sqrtN)
                if c > 0:
                    curr_adj.append((c-1) + r*sqrtN)
                if c < (sqrtN-1):
                    curr_adj.append((c+1) + r*sqrtN)
                adj_list[currVox] = list(np.array(curr_adj))

        if self.kind == 'square':
            z = np.array([
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8])
        elif self.kind == 'ell':
            z = np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        elif self.kind == 'stripes':
            z = np.array([
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        elif self.kind == 'face':
            z = np.asarray([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 3, 3, 3, 1, 1, 1, 1,
                0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 3, 3, 3, 1, 1, 1, 1,
                0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 3, 3, 3, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 1, 1, 1,
                0, 0, 0, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 1, 1, 1,
                0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1,
                0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1,
                0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.z_ = z
        self.adj_list_ = adj_list

    def synthetic_features(self, z):
        """
        Sample synthetic features for a given parcellation.def

        Parameters:
        - - - - -
            z : parcellation
        """

        parcels = {k: np.where(z == k)[0] for k in set(z)}
        params = {k: {'mu': None, 'std': None} for k in set(z)}

        parcel_features = {}.fromkeys(parcels.keys())

        for parc, idx in parcels.items():

            m, s = self.sample_priors()

            params[parc]['mu'] = m
            params[parc]['std'] = s
            parcel_features[parc] = self.sample_features(m, s, len(idx))

        feature_array = np.zeros((len(z), self.d))
        for parc, idx in parcels.items():
            feature_array[idx, :] = parcel_features[parc]

        return [parcels, params, parcel_features, feature_array]

    def sample_priors(self):
        """
        Sample prior mean and variance of synthetic data.

        Parameters:
        - - - - -
            mu_0, kappa_0 : hyperparameters on prior mean
            nu_0, sigma_0 : hyperparameters on prior variance
        """

        sig = invchi2(self.nu0, self.sigma0, self.d)
        mu = norm.rvs(loc=self.mu0, scale=(np.sqrt(sig)/self.kappa0))

        return [mu, sig]

    def sample_features(self, mu, sigma, d):
        """
        Sample from the prior distribution.

        Parameters:
        - - - - -
            mu : prior mean
            sigma : priovar (co)variance
            d : number of samples
        """

        samples = multivariate_normal.rvs(
            mean=mu, cov=np.diag(sigma), size=[d, ])

        return samples


def invchi2(nu, tau2, size):
    """
    Sample from an inverse-chi-squared distribution.
    """

    return invgamma(nu/2, (nu*tau2/2)).rvs(size=size)


def plot_synthetic(synth_object, crp_model, cmap='viridis', figsize=(12, 8)):

    """
    Method to generate reproducible plots of synthetic model performance.

    Parameters:
    - - - - -
    synth_object: SampleSynthetic
        object used to generate synthetic data
    crp_model: ddCRP
        ddCRP model used to fit the clusters to the data
    """

    gt = synth_object.z_.reshape(18, 18)
    mapz = crp_model.map_z_.reshape(18, 18)
    init = crp_model.init_z.reshape(18, 18)

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(
        2, 3, figsize=figsize)

    ax1.imshow(gt, cmap=cmap)
    ax1.set_title('Ground Truth Map', fontsize=14)
    ax2.imshow(mapz, cmap=cmap)
    ax2.set_title('Max-Posteriori Map', fontsize=14)
    ax3.imshow(init, cmap=cmap)
    ax3.set_title('Initialization Map', fontsize=14)

    ax4.plot(crp_model.stats_['K'])
    ax4.set_title('Cluster Count', fontsize=14)
    ax4.set_xlabel('MCMC Iteration', fontsize=14)
    ax4.set_ylabel('Clusters', fontsize=14)

    ax5.plot(crp_model.stats_['lp'])
    ax5.set_title('Log-Probability', fontsize=14)
    ax5.set_xlabel('MCMC Iteration', fontsize=14)
    ax5.set_ylabel('lp', fontsize=14)

    ax6.plot(crp_model.stats_['max_lp'])
    ax6.set_title('Max Log-Probability', fontsize=14)
    ax6.set_xlabel('MCMC Iteration', fontsize=14)
    ax6.set_ylabel('max-lp', fontsize=14)

    plt.tight_layout()
    plt.close()

    return fig
