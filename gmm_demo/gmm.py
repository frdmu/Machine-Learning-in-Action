import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

DEBUG = True

"""
debug output function
"""
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)

"""
Gaussian distribution density function of the k-th model;
i row's data represents the probability of the i-th sample in each model;
return a one-dimensional list;
"""
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

"""
E-step: calculate the expectation of each sample in different model 
"""
def getExpectation(Y, mu, cov, alpha):
    # sample's number 
    N = Y.shape[0]
    # Gause model's number
    K = alpha.shape[0]
    
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # expectation matrix 
    gamma = np.mat(np.zeros((N, K)))
    # calculate each sample's probability in every models
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # calculate each sample's expectation in every models
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

"""
M-step: iterate model's parameters
Y: sample data
gamma: expectation matrix
"""
def maximize(Y, gamma):
    # N:sample's number, D:feature's number 
    N, D = Y.shape
    # model's number
    K = gamma.shape[1]

    alpha = np.zeros(K) 
    mu = np.zeros((K, D))
    cov = []

    # update model's parameters
    for k in range(K):
        # the sum of the expectation of k-th model to all samples
        Nk = np.sum(gamma[:, k])
        # update alpha
        alpha[k] = Nk / N
        # update mu
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        #update cov
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
    cov = np.array(cov)
     
    return mu, cov, alpha


"""  
data preprocessing;
scale all datYa to between 0 and 1
"""
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


"""
initialize model parameters;
shape: sample's shape
K: Gause model's number
"""
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


"""
GMM's EM algorithmY;
Y: sample data
K: Gause model's number
times: number of iteration 
"""
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    # random initialize parameters
    mu, cov, alpha = init_params(Y.shape, K)
    
    for i in range(times):
        # E-step 
        gamma = getExpectation(Y, mu, cov, alpha)
        # M-step 
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


