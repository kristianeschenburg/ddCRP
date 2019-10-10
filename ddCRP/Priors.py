import numpy as np
from numpy.linalg import det
from scipy.special import gammaln, multigammaln
from ddCRP.PriorBase import Prior


class NIW(Prior):

    """
    Normal-Inverse-Chi-Squared prior model for connectivity features.

    Parameters:
    - - - - - -
    mu0, kappa0: float
        priors on distribution mean
    nu0, lambda0: float, array
        priors on distribution variance
    """

    def __init__(self, mu0, kappa0, nu0, lambda0):

        [d, _] = lambda0.shape
        assert nu0 > (d-1), "Degrees of freedom must be greater than the dimension-1."

        self.mu0 = mu0
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.lambda0 = lambda0

    @staticmethod
    def sufficient_statistics(features):
        """
        Compute sufficient statistics for data.

        Parameters:
        - - - - -
        features : float, array
            data array for single cluster

        Returns:
        - - - -
        n : int
            sample size
        mu : array
            mean of each feature
        ssq : array
            sum of squares of each feature
        """
        # n samples
        [n, _] = features.shape
        # feature means
        mu = features.mean(0)
        # scatter matrix
        S = features.T - mu[:, None]
        S = S.dot(S.T)

        return [float(n), mu, S]

    def posterior_parameters(self, suff_stats):
        """
        Computes cluster-specific marginal likelihood hyperparameters
        of a Normal / Normal-Inverse-Chi-Squared model.
        Parameters:
        - - - - -
            suff_stats : sufficient statistics for single cluster
        Returns:
        - - - -
            kappaN : updated kappa
            nuN : updated nu
            sigmaN : updated sigma
        """
        # extract sufficient statistics
        n, mu, S = suff_stats[0:3]

        # update kappa and nu
        kappaN = self.kappa0 + n
        nuN = self.nu0 + n

        central = mu - self.mu0
        scatter = central.dot(central.T)

        deviation = ((n*self.kappa0) / (n+self.kappa0)) * scatter
        lambdaN = self.lambda0 + S + deviation

        return [kappaN, nuN, lambdaN]

    def marginal_evidence(self, posteriors, suff_stats):

        """
        Compute the log-likelihood of the data.

        Parameters:
        - - - - -
        posteriors: list, floats
            posterior hyperparameters
        suff_stats: list, floats
            sufficient statistics
        """

        kappa, nu, L = posteriors[0:3]
        _, p = L.shape
        n = suff_stats[0]

        numer = multigammaln(nu/2, p) + \
            (self.nu0/2)*np.log(np.abs(det(self.lambda0))) + \
            (p/2)*np.log(self.kappa0) 

        denom = multigammaln(self.nu0/2, p) + \
            (nu/2)*np.log(np.abs(det(L))) + \
            (p/2)*np.log(kappa) + \
            (n*p/2)*np.log(np.pi)

        lp = numer - denom

        return lp

    def full_evidence(self, parcels, features):

        """
        Compute the full marginal evidence of a given clustering.

        Parameters:
        - - - - - 
        parcels: dictionary
            mapping of cluster labels to sample indices
        features: float, array
            data feature vectors
        
        Returns:
        - - - -
        lp: float
            full evidence of model
        """

        feats = [features[idx, :] for idx in parcels.values()]
        suff_stats = map(self.sufficient_statistics, feats)
        posteriors = map(self.posterior_parameters, suff_stats)
        cluster_prob = map(self.marginal_evidence, posteriors, suff_stats)

        lp = np.sum(list(cluster_prob))

        return lp



class NIX2(Prior):

    """
    Normal-Inverse-Chi-Squared prior model for connectivity features.

    Parameters:
    - - - - - -
    mu0, kappa0: float
        priors on distribution mean
    nu0, sigma0: float
        priors on distribution variance
    """

    def __init__(self, mu0, kappa0, nu0, sigma0):

        self.mu0 = mu0
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.sigma0 = sigma0

    @staticmethod
    def sufficient_statistics(features):
        """
        Compute sufficient statistics for data.

        Parameters:
        - - - - -
        features : float, array
            data array for single cluster

        Returns:
        - - - -
        n : int
            sample size
        mu : array
            mean of each feature
        ssq : array
            sum of squares of each feature
        """
        # n samples
        [n, _] = features.shape
        # feature means
        mu = features.mean(0)
        # feature sum of squares
        ssq = ((features-mu[None, :])**2).sum(0)

        return [float(n), mu, ssq]

    def posterior_parameters(self, suff_stats):
        """
        Computes cluster-specific marginal likelihood hyperparameters
        of a Normal / Normal-Inverse-Chi-Squared model.
        Parameters:
        - - - - -
            suff_stats : sufficient statistics for single cluster
        Returns:
        - - - -
            kappaN : updated kappa
            nuN : updated nu
            sigmaN : updated sigma
        """
        # extract sufficient statistics
        n, mu, ssq = suff_stats[0:3]

        # update kappa and nu
        kappaN = self.kappa0 + n
        nuN = self.nu0 + n

        deviation = ((n*self.kappa0) / (n+self.kappa0)) * ((self.mu0 - mu)**2)
        sigmaN = (1./nuN) * (self.nu0*self.sigma0 + ssq + deviation)

        return [kappaN, nuN, sigmaN]

    def marginal_evidence(self, posteriors, suff_stats):

        """
        Compute the log-likelihood of the data.

        Parameters:
        - - - - -
        posteriors: list, floats
            posterior hyperparameters
        suff_stats: list, floats
            sufficient statistics
        """

        kappa, nu, sigma = posteriors[0:3]
        p = len(sigma)
        n = suff_stats[0]

        # ratio of gamma functions
        gam = gammaln(nu/2) - gammaln(self.nu0/2)

        # terms with square roots in likelihood function
        inner = (1./2) * (np.log(self.kappa0) + self.nu0*np.log(
            self.nu0*self.sigma0) - np.log(kappa) -
            nu*np.log(nu) - n*np.log(np.pi))

        # sum of sigma_n for each feature
        outer = (-nu/2.)*np.log(sigma).sum()

        lp = p*(gam + inner) + outer

        return lp

    def full_evidence(self, parcels, features):
        """
        Compute the full marginal evidence of a given clustering.

        Parameters:
        - - - - - 
        parcels: dictionary
            mapping of cluster labels to sample indices
        features: float, array
            data feature vectors
        
        Returns:
        - - - -
        lp: float
            full evidence of model
        """

        feats = [features[idx, :] for idx in parcels.values()]
        suff_stats = map(self.sufficient_statistics, feats)
        posteriors = map(self.posterior_parameters, suff_stats)
        cluster_prob = map(self.marginal_evidence, posteriors, suff_stats)

        lp = np.sum(list(cluster_prob))

        return lp
