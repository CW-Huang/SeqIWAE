# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:13:03 2017

@author: Chin-Wei

Sequentialized Sampling Importance Resampling and IWAE training wrapper
"""


import numpy as np
from utils import log_sum_exp_np



def seq_iwae_update(train_func,eval_func,x,spls):
    # use external stochasticity 
    
    """
    Sequentialized IWAE udpate
    
    note that eval_func is for each data point, arguments: input and samples
    spls: batchsize - number of iw samples - dimensions
    """

    weight_func = lambda spl: - eval_func(x,spl)
    re_sample = ssir(weight_func,spls)
    train_func(re_sample)


def ssir(weight_func,samples):
    """
    Sequentialized Sampling Importance Resampling
    
    eval_func: function
    samples: bs-n_iw-dim
    """
    
    acc_weight = - np.inf * np.ones((samples.shape[0]))
    old_spl = np.ones((samples.shape[0],samples.shape[2]))
    for i in range(samples.shape[1]):
        acc_weight, old_spl = refine(weight_func,
                                     acc_weight,
                                     old_spl,
                                     samples[:,i])
    
    return old_spl

    
    

def refine(weight_func,acc_weight,old_spl,new_spl):
    """
    weight_func: function for calculating importance weight on `log` scale,
                    assumed independent accross instances
    acc_weight: accumulated (sum of) importance weight
    old_spl, new_spl: 1d vector of samples
    """
    weights = weight_func(new_spl)
    new_acc_weight = log_sum_exp_np(np.concatenate([acc_weight[:,None],
                                                    weights[:,None]],
                                                   axis=1),axis=1)[:,0]
    
    log_proba_new = weights - new_acc_weight
    log_unif = np.log(np.random.uniform(size=new_spl.shape[0]))
    get_new = (log_proba_new >= log_unif)[:,None]
    condlist = [np.logical_not(get_new), get_new]
    choicelist = [old_spl, new_spl]
    selected_spl = np.select(condlist, choicelist)
    return new_acc_weight, selected_spl



if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    if 0:
        sigmoid = lambda x: 1/(1+np.exp(-x))
        def U(Z):
            z1 = Z[:, 0]
            z2 = Z[:, 1]
            w1 = np.sin(2.*np.pi*z1/5.)
            w3 = 3. * sigmoid((z1-1.)/0.3)
            
            B1 = np.exp(-.5*(((z2-w1)/0.4)**2)+10)
            B2 = np.exp(-.5*(((z2-w1+w3)/0.35)**2)+10)
            B3 = np.exp(-.5*(z1**2 + z2**2/5.)+10)
            return np.log( B1 + B2 + B3 )
        
        sigma = 10. # sample from gaussian proposal with specified std
        n_iw = 20
        
        samples = np.random.randn(1000,n_iw,2) * sigma
        weight_func = lambda x: U(x) - ((x/sigma) ** 2).sum(1)
        
        re_samples = ssir(weight_func,samples)
        
        plt.figure()
        plt.scatter(re_samples[:,0],re_samples[:,1])

    if 1:
        sigmoid = lambda x: 1/(1+np.exp(-x))
        def U2(Z):
            z1 = Z[:, 0]
            z2 = Z[:, 1]
            B1 = 0.30 * np.exp(-(0.2 * z1 - 0.2 * z2 + 2.5)**2)
            B2 = 0.50 * np.exp(-(4.0 * sigmoid(z2) + 2.5 * z1 - 5.0)**2)
            B3 = 0.20 * np.exp(-(0.5 * z1 + 0.5 * z2 + 1)**2 + 0.5 * z1 * z2)
            
            return np.log( B1 + B2 + B3 )
        
        sigma = 10. # sample from gaussian proposal with specified std
        n_iw = 50
        
        samples = np.random.randn(1000,n_iw,2) * sigma
        weight_func = lambda x: U2(x) - ((x/sigma) ** 2).sum(1)
        
        re_samples = ssir(weight_func,samples)
        
        plt.figure()
        plt.scatter(re_samples[:,0],re_samples[:,1])
        plt.xlim(-25,15)
        plt.ylim(-18,22)
    

