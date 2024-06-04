from utils.GMM import*
from spectral_clustering import*
from utils.Plots.matplot_funcs import*
from sklearn import datasets
from keras.src.datasets import mnist
import matplotlib.colors as mcolors
import time
import random

'''NOTE : everything here is broken, as the way the calls to spectral_clustering works have changed. '''


def test_spectral_clustering_circles(k,n_eig,l,g_method,sym_method,sigma):
    noisy_circles = datasets.make_circles(
    n_samples=500, factor=0.5, noise=0.05, random_state=30)
    data,labels=noisy_circles
    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma)

    return data, labels_spectral, labels

def test_spectral_clustering_moons(k,n_eig,l,g_method,sym_method,sigma):
    noisy_moons = datasets.make_moons(
    n_samples=500, noise=0.05, random_state=30)
    data,labels=noisy_moons
    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma)

    return data, labels_spectral, labels

def test_spectral_clustering_gmm(k,n_eig,g_method,sym_method,sigma):
    data,labels=random_GMM(3)
    labels_spectral=[]
    titles=["Unnormalized Laplacian","L_sym","Random walk Laplacian","Ground Truth"]

    for l in ['un_norm','sym','rw']:
        labels_spectral.append(spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma)[1])
    labels_spectral.append(labels)

    return data, labels_spectral, titles

def eigengaps_gmm(k,n_eig,g_method,sym_method,sigma):
    data,labels=random_GMM(4)#To fix
    values=[]
    titles=["L","L_sym","L_rw"]

    for l in ['un_norm','sym','rw']:
        values.append(spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma)[0])

    return values, titles

def eigenboxplot_gmm(k,n_clusters,g_method,sym_method,sigma,n_it):
    settings_title=f"""Boxplots of eigengaps for spectral clustering on randomly generated GMM.
    {n_it} iterations with k = {k},sigma = {sigma}, graph : {g_method}, symmetrizing method : {sym_method}"""
    titles=["L","L_sym","L_rw"]
    Y=[[[] for i in range(n_clusters+1)] for j in range(3)]
    

    for i in range(n_it):
        for j,l in enumerate(["un_norm","sym","rw"]):
            data, labels=random_GMM(n_clusters)
            vals=spectral_clustering(data,k,n_clusters+1,l,g_method,sym_method,sigma, eigen_only=True)
            for k,val in enumerate(vals):
                Y[j][k].append(val)
        print(f" Iterations {(i/n_it)*100}% done")
    return settings_title, titles, Y

def test_sc_single(k,n_clusters,n_eig,l,g_method,sym_method,sigma):
    data,labels=random_GMM(n_clusters)
    values,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma)
    return data,labels_spectral,labels

def spectra(n_it,n_samples,k,n_clusters,l,g_method,sym_method,sigma,n_eig=None):
    if n_eig==None:
        n_eig=n_samples-1
    Y=[[] for i in range(n_eig)]
    title=f"Spectra of L_{l}. Parameters : k = {k}, g_method = {g_method}, sym_method = {sym_method}"
    for i in range(n_it):
        data,labels=random_GMM(n_clusters,n_samples)
        values=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,eigen_only=True)
        for k,val in enumerate(values):
                Y[k].append(val)
        print(f" Iterations {((i+1)/n_it)*100}% done")
    return Y,title

def sc_mnist(k,n_eig,l,g_method,sym_method,sigma):
    (train_data,train_labels),(test_data,test_labels)=mnist.load_data()
    #Reshape the train data (28*28 to 784)
    X = test_data.reshape(len(test_data),-1)
    #Normalize data : [0;255] -> [0;1]
    X=X.astype(float) / 255
    vals,labels_spectral=spectral_clustering(X,k,n_eig,l,g_method,sym_method,sigma,use_minibatch=True,labels_given=test_labels)
    precision=round(len([label for i,label in enumerate(test_labels) if labels_spectral[i]==label])/len(test_labels),4)
    print("Precision of clustering : ",precision*100,"%")
    with open("mnist.txt",'w') as f:
        f.write(f"True Labels : {test_labels} \n Clustering Labels : {labels_spectral}")

def sc_sklearn_datasets(dataset,k,n_eig,l,g_method,sym_method,sigma):

    if dataset=='iris':
        data,labels=datasets.load_iris(return_X_y=True)
    if dataset=='wine':
        data,labels=datasets.load_wine(return_X_y=True)
    if dataset=='breast_cancer':
        data,labels=datasets.load_breast_cancer(return_X_y=True)

    vals,labels_spectral=spectral_clustering(data,k,n_eig,l,g_method,sym_method,sigma,use_minibatch=True,labels_given=labels)
    precision=round(len([label for i,label in enumerate(labels) if labels_spectral[i]==label])/len(labels),4)
    print(f"Dataset: {dataset} \n Settings : L_{l}, k = {k}, g_method={g_method}, sym_method= {sym_method}, sigma = {sigma} \n Accuracy : {precision*100} %")


