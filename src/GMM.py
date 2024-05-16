from misc import*
from matplotlib import cm
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy


class Gaussian:
    """Multivariate gaussian distribution."""
    def __init__(self, d : int,mean ,cov) -> None:
        """
        Inputs :
            d: dimension of the gaussians.
            means (ndarray) : d-dimensional mean.
            cov (ndarray)) : list of the d*d covariance matrices (same size as means)."""
        self.dim=d
        self.mean=mean
        self.cov_matrix=cov
        self.points=np.array([])

    def pdf(self, x):
        '''DOESNT WORK'''
        #Returns the value of the pdf of the distribution for the vector x.
        det=np.linalg.det(self.cov_matrix)
        cov_inv=np.linalg.inv(self.cov_matrix)
        sub=np.subtract(x, self.mean)
        exponent=np.matmul(sub, np.matmul(cov_inv, np.transpose(sub)))
        density=(1/np.sqrt(det * ((2*np.pi)**self.dim))) * np.exp(-(1/2)*exponent)
        return density
    
    def sample(self, N: int):
        """Samples N points from the gaussian distribution.
    
        Inputs :
            N : number of samples
        Returns :
            samples (ndarray) : the sampled points"""
        samples=np.random.default_rng().multivariate_normal(self.mean,self.cov_matrix, N)
        points=self.points.tolist()
        points.append(samples)
        self.points=np.array(points)

        return samples


def GMM(d,N,means,covs):
    """Samples N points from a GMM with dimension d. For each point, the gaussians are uniformly chosen. Also labels points
    according to the gaussian they come from
    
    Inputs :
        d: dimension of the gaussians
        N : number of samples
        means (list) : list of the d-dimensional means
        covs (list of ndarrays) : list of the d*d covariance matrices (same length as means)
    
    Returns :
        samples (ndarray) : the sampled points
        labels (ndarray) : the labels of the sampled points"""
    k=len(means)
    gaussians=[]
    samples=[]
    labels=[]
    for i in range(k):
        gaussians.append(Gaussian(d,means[i],covs[i]))

    for i in range(N):
        r=np.random.default_rng().integers(k)
        samples+=list(gaussians[r].sample(1))
        labels.append(r)

    return np.array(samples), reorder_labels(np.array(means),np.array(labels))


#Some datasets made with gaussians


        
def random_GMM(n_clusters,n_samples_fixed=False):
    if not n_samples_fixed:
        n_samples=random.randint(100,600)
    else:
        n_samples=n_samples_fixed

    means=[]
    for i in range(n_clusters):
        means.append([random.uniform(-10,10),random.uniform(-10,10)])
    covs_initial=[np.add(np.random.default_rng().random(size=(2,2))*(3-0.2), 0.2*np.ones((2,2)))  for i in range(n_clusters)]
    covs_symmetry=[np.matmul(cov,np.transpose(cov)) for cov in covs_initial]
    data, labels=GMM(2,n_samples,means,covs_symmetry)    
    return data, labels
    

def interesting_gmm():
    means=[[0,0],[3,2],[-4,-2],[-2,5]]
    covs=[1*np.identity(2),1*np.identity(2),1*np.identity(2),1*np.identity(2)]
    data, labels=GMM(2,350,means,covs)    
    return data, labels

def circular_GMM(n_clusters,n_samples,sigma,factor):
    means=[(i*factor*sigma,i*factor*sigma) for i in range(n_clusters)]
    covs=[sigma*np.identity(2) for i in range(n_clusters)]
    return GMM(2,n_samples,means,covs)