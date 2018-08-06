import numpy as np
from scipy import stats


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

    def __init__(self, kind, d, mu_0, kappa_0, nu_0, sigma_0):

        self.kind = kind
        self.d = d
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.sigma_0 = sigma_0

    def fit(self):

        """
        Fit prior models and generate synthetic data.
        """

        self._synthetic_labels()
        [pcs, prms, pf, f] = self.synthetic_features(self.synth_.z_)

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
        adj_list = np.empty(sqrtN**2, dtype=object)
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

        x = stats.chi2.rvs(self.nu_0, size=[self.d, ])
        sigma_p = (self.nu_0*self.sigma_0)/x
        mu_p = stats.norm.rvs(self.mu_0, (sigma_p/self.kappa_0))

        return [mu_p, sigma_p]

    def sample_features(self, mu, sigma, d):
        """
        Sample from the prior distribution.

        Parameters:
        - - - - -
            mu : prior mean
            sigma : priovar (co)variance
            d : number of samples
        """

        samples = stats.multivariate_normal.rvs(mean=mu,
                                                cov=np.diag(sigma),
                                                size=[d, ])

        return samples
