from utils.GMM import*
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
        gsc_params=(1,0.7,0.9)
        values=eigengaps_basic('circles',3,4,gsc_params,g_method='knn',sigma=1/2)
        plot_eigenvalues(values,[])'''
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

def gsc_gmm_graph(N,means,covs,l,gsc_params):

    """N=200
    means=[(0,0),(2,2)]
    covs=[1/3*np.identity(2),np.array([[1/3, 1/4],
                                    [1/4,1/3]])]
    gsc_params=(1,0.7,0.9)
    gsc_gmm_graph(N,means,covs,'g_rw',gsc_params)"""
    X,labels=GMM(2,N,means,covs)
    vals,labels_spectral,matrix=spectral_clustering(X,6,4,l,'knn',sym_method=None,gsc_params=gsc_params,sigma=1/3,clusters_fixed=2,return_matrix=True,labels_given=labels)
    plot_sc_graph(X,labels,labels_spectral,matrix)

