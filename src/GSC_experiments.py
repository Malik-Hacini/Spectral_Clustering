from GMM import*
from spectral_clustering import*
from Plots.matplot_funcs import*
from sklearn import datasets
import time
import random


def test_gsc_circles(k,n_eig,l,g_method,sym_method,sigma,gsc_params):
    '''FIG 1 : kmeans vs SC on circles dataset. 
    Plot :
    data,labels_spectral,labels_kmeans,labels=test_spectral_clustering_circles(10,5,'sym','knn','mean',1/8)
    plot_fig1(data,labels_spectral,labels_kmeans)
    '''
    noisy_circles = datasets.make_circles(
    n_samples=300, factor=0.5, noise=0.05, random_state=30)
    data,labels=noisy_circles
    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,gsc_params=gsc_params,clusters_fixed=2)
    labels_kmeans=kmeans(data,2)
    return data, labels_spectral, labels_kmeans, labels

def eigengaps_basic(choice,k,n_eig,gsc_params,g_method='knn',sigma=1/2):
    print(sigma)
    '''FIG E1 and E2
        
        titles,values=eigengaps_basic('circles',7,4)
    plot_eigenvalues(values,titles)'''
    if choice=='circles':
        noisy_circles = datasets.make_circles(
        n_samples=500, factor=0.5, noise=0.05, random_state=30)
        data,labels=noisy_circles
    if choice=='moons':
        noisy_moons = datasets.make_moons(
        n_samples=500, noise=0.05, random_state=30)
        data,labels=noisy_moons

    values=[]

    for l in ['g','g_rw']:
        values.append(spectral_clustering(data,k,n_eig,l,g_method,sigma=sigma,gsc_params=gsc_params,eigen_only=True))
    return values

gsc_params=(1,0.7,0.9)
values=eigengaps_basic('circles',3,4,gsc_params,g_method='knn',sigma=1/2)
plot_eigenvalues(values,[])