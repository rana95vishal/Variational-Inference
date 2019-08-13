# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:24:25 2018

@author: Vishal
"""

#VI for Bi-variate Gaussian 

import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture



#plot_results function has been used from the documentation and examples of scikit learn

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-5., 5.)
    plt.ylim(-5., 5.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500
np.random.seed(1)

#Generating some synthetic data 
num_components = 5
mu_arr = np.array([[-2.5,2.5],[-1.5,1],[3,-1],[1,1],[-3,-1]])
n_sample = 1000
rho_arr = np.array([[[0.1,0],[0,0.7]],[[0.7,-0.2],[-0.2,0.2]],[[0.6,0.1],[0.1,0.4]],[[0.1,0],[0,0.1]],[[0.5,0],[0,0.5]]])
X = np.random.multivariate_normal(mu_arr[0,:],rho_arr[0,:,:], n_sample)
for i in range(num_components-1):
    X = np.append(X,np.random.multivariate_normal(mu_arr[i+1,:],rho_arr[i+1,:,:], n_sample),axis=0)


#average log-likelihood of the data
log_like = np.zeros(45)
for i in range(45):
    
    #VI with random itialization
    #This function uses Dirichlet prior
    dpgmm2 = mixture.BayesianGaussianMixture(n_components=5, max_iter = i+1,init_params = 'random',
                                        covariance_type='full').fit(X)
    if(i%10 == 0):
        plot_results(X, dpgmm2.predict(X), dpgmm2.means_, dpgmm2.covariances_, 1,
             'Bayesian Mixture of Gaussians')
        plt.show()
    log_like[i] = (dpgmm2.score(X[:,:]))
  


plt.plot(log_like, color='b', lw=2.0, label='Average log-likelihood')
plt.title('Bivariate Mixture of Gaussians')
plt.xlabel('iterations'); plt.ylabel('Average log-likelihood')
plt.legend(loc='upper left')
plt.figure()
#plt.savefig('F:/Courses/Computational Statistics/log_like.png')

#VI with kmeans for itialization
dpgmm = mixture.BayesianGaussianMixture(n_components=5, max_iter = 10,init_params = 'kmeans',
                                        covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with kmeans for intialization')
plt.show()
