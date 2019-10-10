import numpy as np

def label_likelihood(label, features, model):

    """
    Compute the log-likelihood of a clustering.

    Parameters:
    - - - - - 
    label: int, array
        clustering results of samples
    features: float, array
        feature vector for each sample
    model: Priors.Prior object
        prior model of feature data
    """

    # get unique label values
    u_labs = np.unique(label)
    u_labs = u_labs[u_labs > 0]

    parcels = {}
    for lab in u_labs:
        parcels[lab] = np.where(label == lab)[0]

    fp = fullProbability(parcels, features, model)

    return fp

def fullProbability(parcels, features, model):

    """
    Compute the full log-likelihood of the clustering.
    Parameters:
    - - - - -
    parcels : dictionary
            mapping between cluster ID and sample indices
    features : array
            data samples
    Returns:
    - - - -
    lp : float
            marginal log-likelihood of a whole parcelation
    """

    feats = [features[idx, :] for idx in parcels.values()]

    suff_stats = map(model.sufficient_statistics, feats)
    posteriors = map(model.posterior_parameters, suff_stats)
    cluster_prob = map(model.marginal_evidence, posteriors, suff_stats)

    lp = np.sum(list(cluster_prob))

    return lp
