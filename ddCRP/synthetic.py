import numpy as np
import collections
from scipy import stats

# Format of generated synthetic datasets
SynthData = collections.namedtuple('SynthData',['D','adj_list','z','coords'])

# Generate synthetic dataset of "type"={'square','stripes','face'} at a given
#   noise level "sig". Returns a SynthData object containing a connectivity
#   matrix D, and adjacency list adj_list, ground truth parcellation z, and
#   element coordinates coords
def GenerateSynthData(type, sig):
    sqrtN = 18

    coords = np.zeros((sqrtN**2,2))
    adj_list = np.empty(sqrtN**2, dtype=object)
    for r in range(0, sqrtN):
        for c in range(0, sqrtN):
            currVox = c + r*sqrtN
            coords[currVox,:] = [r, c]
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
    
    if type == 'square':
        z = np.array([
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8])
    elif type == 'ell':
        z = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
    elif type == 'stripes':
        z = np.array([
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
    elif type == 'face':
        z = np.asarray([
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,2,2,2,0,0,1,1,3,3,3,1,1,1,1,
        0,0,0,0,2,2,2,0,0,1,1,3,3,3,1,1,1,1,
        0,0,0,0,2,2,2,0,0,1,1,3,3,3,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,4,4,0,0,0,0,1,1,1,1,4,4,1,1,1,
        0,0,0,4,4,0,0,0,0,1,1,1,1,4,4,1,1,1,
        0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,
        0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,
        0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]);
        
    N = len(z)
    K = len(np.unique(z))
    
    A = np.random.normal(size=(K,K))
    D = np.zeros((N,N))

    for v1 in range(0,N):
        for v2 in range(0,N):
            if v1 != v2:
                D[v1,v2] = sig*np.random.normal() + A[z[v1],z[v2]]
    
    synth = SynthData(D, adj_list, z, coords)
    return synth

def GenerateSynthFeatures(z,d,mu_0,kappa_0,nu_0,sigma_0):

    """
    Sample synthetic features for a given parcellation.def

    Parameters:
    - - - - -
        z : parcellation
        d : feature dimensions
        mu_0, kappa_0 : hyperparameters on prior mean
        nu_0, sigma_0 : hyperparameters on prior variance
    """

    parcels = {k : np.where(z == k)[0] for k in set(z)}
    params = {k: {'mu': None, 'std': None} for k in set(z)}

    parcel_features = {}.fromkeys(parcels.keys())

    for parc,idx in parcels.items():

        m,s = sample_priors(mu_0,kappa_0,nu_0,sigma_0,d)
        params[parc]['mu'] = m
        params[parc]['std'] = s

        parcel_features[parc] = sample_features(m,s,len(parcels[parc]))

    feature_array = np.zeros((len(z),d))
    for parc,idx in parcels.items():
        feature_array[idx,:] = parcel_features[parc]


    return [parcels,params,parcel_features,feature_array]

def sample_priors(mu_0,kappa_0,nu_0,sigma_0,size):

    """
    Sample prior mean and variance of synthetic data.

    Parameters:
    - - - - -
        mu_0, kappa_0 : hyperparameters on prior mean
        nu_0, sigma_0 : hyperparameters on prior variance
    """

    x = stats.chi2.rvs(nu_0,size=[size,])
    sigma_p = (nu_0*sigma_0)/x;
    mu_p = stats.norm.rvs(mu_0,(sigma_p/kappa_0))

    return [mu_p,sigma_p]

def sample_features(mu,sigma,d):

    """
    Sample from the prior distribution.

    Parameters:
    - - - - -
        mu : prior mean
        sigma : priovar (co)variance
        d : number of samples
    """

    samples = stats.multivariate_normal.rvs(mean=mu,cov=np.diag(sigma),size=[d,])

    return samples