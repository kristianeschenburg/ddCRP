import numpy as np
from numpy import linalg
from scipy.special import gammaln, multigammaln
from PriorBase import Prior


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

    def log_likelihood(self, posteriors, suff_stats):

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

        # ratio of gamma functions
        gam = multigammaln(nu/2, p) - multigammaln(self.nu0/2, p)

        # terms with square root in marginal likelihood
        inner = (1/2) * (
            self.nu0*np.log(linalg.det(self.lambda0)) -
            nu*np.log(linalg.det(L)) +
            p * (
                np.log(self.kappa0) -
                np.log(kappa) -
                n*np.log(np.pi)))

        lp = gam + inner

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

    def log_likelihood(self, posteriors, suff_stats):

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