import numpy as np
from scipy import sparse
from scipy.special import gammaln

from sampling import Gibbs
from subgraphs import ConnectedComponents, ClusterSpanningTrees
import statistics

import time


class ddCRP(object):
    
    """
    Class to implement the distance-dependent Chinese Restaurant Process.
    
    Parameters:
    - - - - -
        alpha : concentration parameter of CRP prior
        mu_0, kappa_0 : hyperparameters on feature mean prior
        nu_0, sigma_0 : hyperparameters on feature variance prior

        mcmc_passes : number of MCMC passes to apply to data
        stats_interval : number of passes to run before recording statistics
        verbose : boolean to print statistics every stats_interval
        
    """
    
    def __init__(self,alpha,mu_0,kappa_0,nu_0,sigma_0,mcmc_passes=100,
        stats_interval=500,verbose=True):
        
        """
        Initialize ddCRP object.
        """
        
        self.alpha = np.float(alpha)
        self.mu0 = np.float(mu_0)
        self.kappa0 = np.float(kappa_0)
        self.nu0 = np.float(nu_0)
        self.sigma0 = np.float(sigma_0)
        self.mcmc_passes = np.int(mcmc_passes)
        self.stats_interval = np.int(stats_interval)
        self.verbose = verbose

    def fit(self,features,adj_list,init_c=None,gt_z=None,edge_prior=None):

        """
        Main function to fit the distance-dependent Chinese Restaurant Process.Restaurant

        Parameters:
        - - - - -
            features : data array of features for each sample
            adj_list : adjacency list of samples
            init_c : initialized cortical map, default = []
            gt_z : ground truth map for computing normalized mutual information
            edge_prior : nested dictionary, probability of neighboring vertices beloning
                        to same parcels

        """

        # initialize Gibbs sampling object
        gibbs = Gibbs()
        nvox = len(adj_list)

        stats = {'times': [],'lp': [],'max_lp': [],
                    'K': [],'z': np.empty((0,nvox)),
                    'c': np.empty((0,nvox)),'NMI': []}

        # initialize parent vector, if not provided
        if not np.any(init_c):
            c = np.zeros((nvox,))
            for i in np.arange(nvox):
                neighbors = adj_list[i] + [i]
                c[i] = neighbors[np.random.randint(low=0,high=len(neighbors))]
        else:
            c = init_c

        c = c.astype(np.int32)

        # initialize sparse linkage matrix
        G = sparse.csc_matrix((np.ones((nvox,)),(np.arange(nvox),c)), shape=(nvox,nvox))
        G = G.tolil()
        
        # compute initial parcel count and parcel assignments
        [K, z, parcels] = ConnectedComponents(G)
        self.init_z = z

        # compute log-likelihood of initial cortical map
        curr_lp = self._fullProbabilityDDCRP(parcels,features)

        max_lp = -1.*np.inf
        map_z = []
        t0 = time.time()
        steps = 0

        order = np.arange(nvox)

        # perform mcmc_passes of over all samples
        for mcmc_pass in np.arange(self.mcmc_passes):

            # shuffle sample order for each MCMC pass
            np.random.shuffle(order)

            for i in order:

                # if current map log-probability greater than current max
                # set current map to best map
                if curr_lp > max_lp:
                    max_lp = curr_lp
                    map_z = z

                if steps % self.stats_interval == 0:
                    stats = statistics.UpdateStats(stats, t0, curr_lp, max_lp,
                                    K, list(z), list(c), steps, gt_z, map_z, self.verbose)

                # remove current link to parent
                G[i,c[i]] = 0

                # if link was self-link
                if c[i] == i:
                    # Removing self-loop, parcellation won't change
                    rem_delta_lp, z_rem, parcels_rem = -np.log(self.alpha), z, parcels
                else:
                    # otherwise compute new connected components
                    K_rem, z_rem, parcels_rem = ConnectedComponents(G)

                    # if number of components changed
                    if K_rem != K:
                        # We split a cluster, compute change in likelihood
                        rem_delta_lp = -self._LogProbDiff(parcels_rem,z_rem[i],z_rem[c[i]],features)

                    else:
                        rem_delta_lp = 0

                # get neighbors of sample i
                adj_list_i = adj_list[i]

                # initialize empty log-prob vector
                lp = np.zeros((len(adj_list_i)+1,))
                lp[-1] = np.log(self.alpha)

                for j,n in enumerate(adj_list_i):
                    # just undoing split
                    if z_rem[n] == z_rem[c[i]]:
                        lp[j] = -rem_delta_lp - (c[i] == i)*np.log(self.alpha)

                    # (possibly) new merge
                    elif z_rem[n] != z_rem[i]:
                        lp[j] = self._LogProbDiff(parcels_rem,z_rem[i],z_rem[n],features)

                # sample new neighbor according to Gibbs
                new_neighbor = gibbs.fit(lp)
                if new_neighbor < len(adj_list_i):
                    c[i] = adj_list_i[new_neighbor]
                else:
                    c[i] = i

                curr_lp = curr_lp + rem_delta_lp + lp[new_neighbor]
                G[i,c[i]] = 1
                [K,z,parcels] = ConnectedComponents(G)
                steps += 1

        stats = statistics.UpdateStats(stats, t0, curr_lp, max_lp,
                    K, list(z), list(c), steps, gt_z, map_z, self.verbose)

        # for visualization purposes
        map_z[np.where(map_z == 0)[0]] = map_z.max() + 1

        self.map_z = map_z
        self.stats = stats

    def _fullProbabilityDDCRP(self,parcels,features):
        
        """
        Compute the full log-likelihood of the clustering.
        
        Parameters:
        - - - - -
            parcels : dictionary mapping cluster IDs to data indices
            features : array of features for full dataset
        Returns:
        - - - -
            lp : marginal log-likelihood of a whole parcelation
        """
        
        lp = 0
        
        for parc,idx in parcels.items():
            
            sufficient = self._sufficient_statistics(features[idx,:])
            params_n = self._marginal_parameters(sufficient)
            lp += self._LikelihoodCluster(params_n,sufficient[0])

        return lp
    
    def _LikelihoodCluster(self,params,n):
        
        """
        Computes the log marginal likelihood of a single cluster using the
        a Normal likelihood and Normal-Inverse-Chi-Squared prior.
        
        Parameters:
        - - - - -
            kappa : kappa of cluster, marginalized of mu0,sigma0
            nu : nu of cluster, marginalized of mu0,sigma0
            sigma : sigma of cluster, marginalized of mu0,sigma0
            n : sample size of cluster
        Returns:
        - - - -
            lp : marginal log-likelhood of a single cluster
        """

        kappa,nu,sigma = params[0:3]
        p = len(sigma)

        # ratio of gamma functions
        gam = gammaln(nu/2) - gammaln(self.nu0/2)

        # terms with square roots in likelihood function
        inner = (1./2) * (np.log(self.kappa0) + self.nu0*np.log(self.nu0*self.sigma0) - 
                         np.log(kappa) - nu*np.log(nu) - n*np.log(np.pi))
        
        # sum of sigma_n for each feature
        outer = (-nu/2.)*np.log(sigma).sum()
        
        lp = p*(gam + inner) + outer
        
        return lp

    def _LogProbDiff(self,parcel_split,split_l1,split_l2,features):

        """
        Compute change in log-likelihood when considering a merge.

        Parameters:
        - - - - -
            parcel_split : indices of each component in split map
            split_l1 , split_l2 : label values of components to merge
            features : input feature array
        Returns:
        - - - -
            ld : log-likelihood difference between merging and splitting 
                    two clusters
        """

        merged_indices = np.concatenate([parcel_split[split_l1],
            parcel_split[split_l2]])

        # compute sufficient statistics and marginalized parameters of merged parcels
        stats = self._sufficient_statistics(features[merged_indices,:])
        phyp = self._marginal_parameters(stats)

        # compute likelihood of merged parcels
        merge_ll = self._LikelihoodCluster(phyp,stats[0])

        # compute likelihood of split parcels
        split_ll = self._LogProbSplit(parcel_split,split_l1,split_l2,features)

        ld = merge_ll - split_ll

        return ld

    def _LogProbSplit(self,parcel_split,split_l1,split_l2,features):

        """
        Compute change in log-likelihood when consiering a split.

        Parameters:
        - - - - -
            parcel_split : indices of each component in split map
            split_l1 , split_l2 : label values of components to merge
            features : input feature array
        Returns:
        - - - -
            split_ll : log-likelihood of two split clusters
        """

        idx1 = parcel_split[split_l1]
        idx2 = parcel_split[split_l2]

        suff1 = self._sufficient_statistics(features[idx1,:])
        suff2 = self._sufficient_statistics(features[idx2,:])

        phyp1 = self._marginal_parameters(suff1)
        phyp2 = self._marginal_parameters(suff2)

        lp_1 = self._LikelihoodCluster(phyp1,suff1[0])
        lp_2 = self._LikelihoodCluster(phyp2,suff2[0])

        split_ll = lp_1 + lp_2

        return split_ll
 
    def _sufficient_statistics(self,cluster_features):
        
        """
        Compute sufficient statistics for data.
        
        Parameters:
        - - - - -
            cluster_features : data array for single cluster 
        Returns:
        - - - -
            n : sample size
            mu : mean of each feature
            ssq : sum of squares of each feature
        """
        
        # n samples
        [n,_] = cluster_features.shape
        # feature means
        mu = cluster_features.mean(0)
        # feature sum of squares
        ssq = ((cluster_features-mu[None,:])**2).sum(0)

        return [float(n),mu,ssq]
    
    def _marginal_parameters(self,suff_stats):
        
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
        n,mu,ssq = suff_stats[0:3]

        # update kappa and nu
        kappaN = self.kappa0 + n
        nuN = self.nu0 + n
        
        deviation = ((n*self.kappa0) / (n+self.kappa0)) * ((self.mu0 - mu)**2)
        sigmaN = (1./nuN) * (self.nu0*self.sigma0 + ssq + deviation)
        
        return [kappaN,nuN,sigmaN]
    