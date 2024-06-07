import pytest
from utils.GMM import*
from utils.data_files_managing import*
from spectral_clustering import*
from utils.Plots.matplot_funcs import*
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

def test_spectral_clustering_moons(k,n_eig,l,g_method,sym_method,sigma):
    '''FIG 2 : kmeans vs SC on moons dataset. 
    Plot :
    
    data,labels_spectral,labels_kmeans,labels=test_spectral_clustering_moons(10,5,'sym','knn','mean',1/8)
    plot_fig1(data,labels_spectral,labels_kmeans)

    '''
    noisy_moons = datasets.make_moons(
    n_samples=500, noise=0.05, random_state=30)
    data,labels=noisy_moons
    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,clusters_fixed=2)
    labels_kmeans=kmeans(data,2)
    return data, labels_spectral, labels_kmeans, labels

def gaussian_mixture(N,means,covs):
    '''FIG 3 :
    N=200
means=[(0,0),(2,2)]
covs=[1/3*np.identity(2),np.array([[1/3, 1/4],
                                   [1/4,1/3]])]

X,labels=gaussian_mixture(N,means,covs)
plot_fig3_binorm(X,labels)'''
    X,labels=GMM(2,N,means,covs)
    return X,labels

def gaussian_mixture_fail():
    'FIG 5'
    data,labels=circular_GMM(2,100,1/3,3.5)
    with open('src/gaussians_data.txt','w') as f:
        data_str=[" ".join(datum) for datum in data.astype(str)]
        f.write("\n".join(data_str))
    with open('src/gaussians_labels.txt','w') as f:
        f.write(" ".join(labels.astype(str)))
    vals,labels_spectral,matrix=spectral_clustering(data,6,4,'rw','knn','mean',1/3,clusters_fixed=2,return_matrix=True,labels_given=labels)
    plot_sc_graph(data,labels,labels_spectral,matrix)

def gaussian_mixture_fail_fixed():
    'FIG 6'
    data,labels=np.loadtxt('src/gaussians_data.txt'),np.loadtxt('src/gaussians_labels.txt')
    vals,labels_spectral,matrix=spectral_clustering(data,8,4,'rw','knn','mean',1/3,clusters_fixed=2,return_matrix=True)
    plot_sc_graph(data,labels,labels_spectral,matrix)

def eigengaps_basic(choice,k,n_eig,g_method='knn',sym_method='mean',sigma=1/2):
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

    for l in ['un_norm','sym','rw']:
        values.append(spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,eigen_only=True))
    
    return values


def eigengap_circ_gmm(n_eig):
    data,labels=circular_GMM(2,100,1/3,3.5)
    with open('src/gaussians_data.txt','w') as f:
        data_str=[" ".join(datum) for datum in data.astype(str)]
        f.write("\n".join(data_str))
    with open('src/gaussians_labels.txt','w') as f:
        f.write(" ".join(labels.astype(str)))
    vals=spectral_clustering(data,8,n_eig,'un_norm','knn','mean',1/3,eigen_only=True)
    X=np.arange(1,n_eig+1,1)
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.locator_params(axis="y", tight=True,nbins=4)
    plt.scatter(X,vals)
    plt.show()


def clustering_gmm(k):
    '''FIG 8 '''
    data,labels=interesting_gmm()
    titles=["L","L_sym","L_rw"]
    vals,labels_spectral,matrix=spectral_clustering(data,k,5,'rw','knn','mean',1/2,return_matrix=True)
    plot_sc_graph(data,labels,labels_spectral,matrix)


def simgraph_example():
    data,labels,name=load_data_n_labels('simgraph_example')
    matrix=spectral_clustering(data,n_clusters=2,k_neighbors=4,return_labels=False,return_eigvals=False,return_matrix=True)[0]
    plot_simgraph(data,matrix)

def clustering_example():
    data,labels,name=load_data_n_labels('simgraph_example')
    labels_spectral,matrix=spectral_clustering(data,n_clusters=2,k_neighbors=4,return_eigvals=False,return_matrix=True)
    plot_simgraph(data,matrix,labels=labels)


def connected_example():
    data,labels,name=load_data_n_labels('connected_example')
    labels_spectral,matrix=spectral_clustering(data,n_clusters=2,k_neighbors=4,return_eigvals=False,return_matrix=True)
    plot_simgraph(data,matrix,labels=labels)


def eigvals_comparison():
    steps=[100,200,500]

    means=[(0,0),(1.15,1.15)]
    n_clusters=len(means)
    distrib=[0.5,0.5]
    sigmas_x=[1/3,1/3]
    sigmas_y=[1/3,1/3]
    p_list=[0,0]
    covs=[bivariate_cov_m(sigmas_x[i],sigmas_y[i],p_list[i]) for i in range(len(means))]
    gsc_params=(1,1,0.99)

    values=[]
    for step in steps:
        data,labels=GMM(n_clusters,step,means,covs,distrib)  

        labels_spectral,vals,matrix=spectral_clustering(data,k_neighbors=6,n_eig=3,laplacian='g_rw',
                                                           sigma=1,n_clusters=2,return_matrix=True)
        values.append(list(vals))
        
    plot_eigengap_comparison(values,steps)


eigvals_comparison()