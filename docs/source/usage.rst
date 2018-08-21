=====
Usage
=====

```python
from ddCRP import ddCRP
from ddCRP import synthetic

# set hyperparameter values
alpha = 10
mu = 0
kappa = 0.0001
nu = 1
sigma = 1

# dimensionality of data
d = 5

# sample synthetic features for each label
# If you want to sample from a different Normal-Inverse-Chi-Squared
# distribution, change kappa, mu, nu, and sigma
synth = synthetic.SampleSynthetic(kind='ell', d, mu, kappa, nu, sigma)
synth.fit()

# fit the ddCRP model
# after fitting, crp.map_z_ is the MAP label
crp = ddCRP.ddCRP(alpha,mu,kappa,nu,sigma,mcmc_passes=30,stats_interval=200)
crp.fit(synth.features_, synth.adj_list, gt_z=synth.z_)
```
