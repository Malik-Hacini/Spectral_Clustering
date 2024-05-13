import pytest
from GMM import*
from spectral_clustering import*
from Plots.plot_funcs import*
from sklearn import datasets
import matplotlib.colors as mcolors
import time
import random

def test_spectral_clustering_circles(k,n_eig,l,g_method,sym_method,sigma):
    '''FIG 1 : kmeans vs SC on circles dataset. 
    Plot :
    data,labels_spectral,labels_kmeans,labels=test_spectral_clustering_circles(10,5,'sym','knn','mean',1/8)
    plot_fig1(data,labels_spectral,labels_kmeans)
    '''
    noisy_circles = datasets.make_circles(
    n_samples=500, factor=0.5, noise=0.05, random_state=30)
    data,labels=noisy_circles
    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,clusters_fixed=2)
    labels_kmeans=kmeans(data,2)
    return data, labels_spectral, labels_kmeans, labels
