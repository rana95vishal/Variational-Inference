# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:06:15 2018

@author: Vishal
"""

# CAVI for 1D Bayesian Mixture of Gaussians

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


np.random.seed(7)
itr_n = 40


N = 3           #number of components
sigma2 = 1      #hyperparameter, variance of the prior on means

M = 1000     #number of points per component
mu_arr = np.random.choice(np.arange(-10, 10, 2),N) + np.random.random(N)  #Generating the means for data

#Generating the data
X = np.random.normal(loc=mu_arr[0], scale=1, size=M)
for i, mu in enumerate(mu_arr[1:]):
    X = np.append(X, np.random.normal(loc=mu, scale=1, size=M))

#declaring and initalizing the parameters and the ELBO
ELBO = np.zeros(itr_n)
phi = np.reshape(np.zeros((M*N)*(N)),(M*N,N))
mu_x = np.random.randint(int(X.min()), high=int(X.max()), size=N).astype(float) + X.max()*np.random.random(N)
sigma_x = np.ones(N) * np.random.random(N)


for j in range(itr_n):
    
    #updating the vales of parameters
    t1 = np.outer(X, mu_x)
    t2 = -(0.5*mu_x**2 + 0.5*sigma_x)
    phi = np.exp(t1 + t2[np.newaxis, :])
    phi = phi / phi.sum(1)[:, np.newaxis]
    
    mu_x = (phi*X[:, np.newaxis]).sum(0) * (1/sigma2 + phi.sum(0))**(-1)
    sigma_x = (1/sigma2 + phi.sum(0))**(-1)
    
    
    #calculating the ELBO
    t1 = (np.log(sigma_x) - mu_x/sigma2)*0.5
    t1 = t1.sum()
    t2 = -0.5*np.add.outer(X**2, mu_x**2)
    t2 += np.outer(X, mu_x)
    t2 -= np.log(phi)
    t2 *= phi
    t2 = t2.sum()
    ELBO[j] = t1+t2

plt.figure()    
plt.title('Bayesian Mixture of Gaussians')
plt.xlabel('x'); plt.ylabel('probability')
plt.hist(X[:M], normed=True, bins=100,color='gold',alpha=0.4)
plt.hist(X[M+1:2*M], normed=True, bins=100,color='darkorange',alpha=0.4)
plt.hist(X[2*M+1:3*M], normed=True, bins=100,color='navy',alpha=0.4)
plt.plot(np.linspace(-12, 12, 100),mlab.normpdf(np.linspace(-12, 12, 100), mu_x[0], 1))
plt.plot(np.linspace(-12, 12, 100),mlab.normpdf(np.linspace(-12, 12, 100), mu_x[1], 1))
plt.plot(np.linspace(-12, 12, 100),mlab.normpdf(np.linspace(-12, 12, 100), mu_x[2], 1))
#plt.savefig('F:/Courses/Computational Statistics/gmm_vi_1d.png')

plt.figure()
plt.title('ELBO for Mixture of Gaussian problem')
plt.xlabel('iterations'); plt.ylabel('ELBO objective')
plt.plot(ELBO, color='b', lw=2.0, label='ELBO')
#plt.savefig('F:/Courses/Computational Statistics/gmm_vi_1d_elbo.png')