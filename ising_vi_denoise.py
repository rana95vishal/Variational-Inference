# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:21:19 2018

@author: Vishal
"""

#Image de-noising using Ising model

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal

np.random.seed(0)
sns.set_style('whitegrid')


#data generating method adopted from ref [6] in report
#loading the image
#please modify the loaction of the file depending on where you have it saved
data = Image.open('F:/Courses/Computational Statistics/kung.bmp')
img_tmp = np.double(data)
img = img_tmp[:,:,0]
#binarizing the image
img_mean = np.mean(img)
img_binary = -1*(img>img_mean) + +1*(img<img_mean)
[M, N] = img_binary.shape


#mean-field parameters
sigma  = 2  #noise level
y = img_binary + sigma*np.random.randn(M, N) #y_i ~ N(x_i; sigma^2);
J = 1  #coupling strength (w_ij) assumed constant
rate = 0.5  #update smoothing rate, lambda
max_iter = 20
ELBO = np.zeros(max_iter)
Hx_mean = np.zeros(max_iter)

#generate plots
plt.figure()
plt.imshow(y)
plt.title("observed noisy image")
#plt.savefig('F:/Courses/Computational Statistics/ising_vi_observed_image.png')

#Mean-Field VI
print ("running mean-field variational inference...")

logp1 = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=+1, cov=sigma**2), (M, N))
logm1 = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=-1, cov=sigma**2), (M, N))
logodds = logp1 - logm1

#initialize
mu = np.reshape(np.zeros((M+2)*(N+2)),(M+2,N+2))
mu[1:M+1,1:N+1] = 2*sigmoid(logodds)-1  #mu_init

a = mu[1:M+1,1:N+1] + 0.5 * logodds
qxp1 = sigmoid(+2*a)  #q_i(x_i=+1)
qxm1 = sigmoid(-2*a)  #q_i(x_i=-1)

for i in (range(max_iter)):
    muNew = mu
    
    #sum of neighbouring states
    m_sum = np.reshape(np.zeros((M)*(N)),(M,N))
    
    for ix in range(1,N+1):
        for iy in range(1,M+1):
            #calculating sum of neighbouring states 
            m_sum[iy-1,ix-1] = J*(mu[iy,ix-1] + mu[iy,ix+1] + mu[iy-1,ix] + mu[iy+1,ix])     
            
            #updating means
            muNew[iy,ix] = (1-rate)*muNew[iy,ix] + rate*np.tanh(m_sum[iy-1,ix-1] + 0.5*logodds[iy-1,ix-1])
            
    mu = muNew
            
    a = mu[1:M+1,1:N+1] + 0.5 * logodds
    qxp1 = sigmoid(+2*a) #q_i(x_i=+1)
    qxm1 = sigmoid(-2*a) #q_i(x_i=-1)    
    Hx = -qxm1*np.log(qxm1+1e-10) - qxp1*np.log(qxp1+1e-10) #entropy        
    
    ELBO[i] = ELBO[i] + np.sum(m_sum*(qxp1 - qxm1)) +np.sum(qxp1*logp1 + qxm1*logm1) + np.sum(Hx)
    Hx_mean[i] = np.mean(Hx) 
    
plt.figure()
plt.imshow(mu)
plt.title("after %d mean-field iterations" %max_iter)
#plt.savefig('F:/Courses/Computational Statistics/ising_vi_denoised_image1.png')

plt.figure()
plt.plot(ELBO, color='b', lw=2.0, label='ELBO')
plt.title('Variational Inference for Ising Model')
plt.xlabel('iterations'); plt.ylabel('ELBO objective')
plt.legend(loc='upper left')
#plt.savefig('F:/Courses/Computational Statistics/ising_vi_elbo1.png')

plt.figure()
plt.plot(Hx_mean, color='b', lw=2.0, label='Avg Entropy')
plt.title('Variational Inference for Ising Model')
plt.xlabel('iterations'); plt.ylabel('average entropy')
plt.legend(loc="upper right")
#plt.savefig('F:/Courses/Computational Statistics/ising_vi_avg_entropy1.png')



