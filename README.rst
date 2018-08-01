===============================
ddCRP
===============================

.. image:: https://img.shields.io/travis/kristianeschenburg/ddCRP.svg
        :target: https://travis-ci.org/kristianeschenburg/ddCRP

.. image:: https://img.shields.io/pypi/v/ddCRP.svg
        :target: https://pypi.python.org/pypi/ddCRP


Package for modeling multivariate, graph-based data via the distance-dependent Chinese Restaurant Process.

This is a package to apply the distance-dependent Chinese Restaurant Process (dd-CRP) to multi-dimensional graph-based data.  It is based roughly on code originally written by Christopher Baldassano (*Baldassano et al. (2015), Parcellating Connectivity In Spatial Maps. PeerJ 3:e784; DOI 10.7717/peerj.784*).

In contrast to work presented by Baldassano et al. 2015 and Moyer et al. 2017, whose methods both model the univariate similarities within and between clusters, this method models the clusters themselves, placing priors on the connectivity features of each cluster.

Example of use on synthetic data:

.. code-block:: python
	from ddcrp import ddCRP
	import synthetic

	# set hyperparameter values
	alpha = 10
	mu = 0
	kappa = 0.0001
	nu = 1
	sigma = 1

	# dimensionality of data
	d = 5

	# generate synthetic label map
	Synth = synthetic.GenerateSynthData(type='ell',sig=0.1)

	# sample synthetic features for each label
	# If you want to sample from a different Normal-Inverse-Chi-Squared
	# distribution, change kappa, mu, nu, and sigma
	[_,_,_,features] = synthetic.GenerateSynthFeatures(Synth.z,d,mu,kappa,nu,sigma)

	# fit the ddcrp model
	# after fitting, crp.map_z is the MAP label
	crp = ddCRP(alpha,mu,kappa,nu,sigma,mcmc_passes=30,stats_interval=200)
	crp.fit(features,Synth.adj_list,init_c=None)


!['Model performance on 'ell' synthetic datasetg.'](https://github.com/kristianeschenburg/ddCRP/ddCRP/figures/ell.jpg)

For more information on the Chinese Restaurant Process, see:

  * Baldassano et al. (2015), Parcellating Connectivity In Spatial Maps. PeerJ 3:e784; DOI 10.7717/peerj.784

  * Moyer et al. (2017), A Restaurant Process Mixture Model for Connectivity Based Parcellation of the Cortex. 	arXiv:1703.00981


* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://kristianeschenburg.github.io/ddCRP.

Features
--------

* TODO
