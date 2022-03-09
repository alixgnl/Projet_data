from sklearn import mixture

import numpy as np
import matplotlib.pyplot as plt

mu_0 = 5.0
srd_0 = 2.0

data = np.random.randn(100000)
data = data * srd_0 + mu_0

data = data.reshape(-1, 1)

hx, hy, _ = plt.hist(data, bins=50, density=1,color="lightblue")

plt.ylim(0.0,max(hx)+0.05)
plt.title('Gaussian mixture example 01')
plt.grid()

plt.xlim(mu_0-4*srd_0,mu_0+4*srd_0)

plt.savefig("example_gmm_01.png", bbox_inches='tight')
plt.show()

gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(data)

print(gmm.means_)
print(np.sqrt(gmm.covariances_))

[[5.00715457]]
[[[1.99746652]]]