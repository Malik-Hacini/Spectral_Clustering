from graphs import*
from utils.labels_ordering import*
from scipy.linalg import eigh,eig,issymmetric
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans


def normalize_vec(vector):
    """Normalizes a vector
    Inputs :
        vector(ndarray): n-d vector.
    
    Returns :
        vec_normalized (ndarray) : n-d vector with norm 1."""
    
    return (1/np.linalg.norm(vector)) * vector

def compute_centers(data,labels):
    clusters=[[datum for j,datum in enumerate(data) if labels[j]==i] for i in range(len(set(labels)))]
    centers=[[sum(i)//len(cluster) for i in zip(*cluster)] for cluster in clusters]
    return centers

def kmeans(data,k,use_minibatch=False):
    '''Performs the k-means clustering algorithm on a dataset of n-dimensional points. 
    Use LLoyd's algorithm to compute the clustering.
    
    Inputs :
        data (ndarray): dataset.
        k (int): Number of clusters (or centroids) to construct.
    
    Returns :
        labels (ndarray) : labels of the points after k-means, ordered in the same way as the dataset.'''
    if use_minibatch:
        est=MiniBatchKMeans(n_clusters=k)
    else:
        est=KMeans(n_clusters=k)
    est.fit(data)

    return est.labels_

def cluster(n_clusters,n_eig,laplacian_matrix,use_minibatch):
    vals,u_full=eigenvectors(n_eig,*laplacian_matrix)
    
    #In the case of L_sym being used, there is an additional normalization step.
    if laplacian_matrix=='sym':
        u_full=np.apply_along_axis(normalize_vec, axis=0, arr=u_full)
    u=u_full[:,:n_clusters]
    print("k-means clustering the spectral embedded data...")
    labels_unord=kmeans(u,n_clusters,use_minibatch)

    return vals,labels_unord


def eigenvectors(i,a,b=None):
    """Computes the first i eigenvals and eigenvecs of a symmetric matrix for the generalized problem
    a*X=lambda*b*X
    Inputs :
        a(ndarray): Needs to be real symmetric.
        i (int): Number of eigenvals and eigenvecs to compute (in ascending order)
        b (ndarray/str) : The second matrix / a string.
    Returns :
        vecs (ndarray) : Matrix with the i eigenvecs as columns."""
    if isinstance(b,str):
        vals,vecs=eig(a)
        vals,vecs=vals.real,vecs.real
        idx = vals.argsort()
        vals = vals[idx]
        vecs = vecs[:,idx]
        vals,vecs=vals[:i],vecs[:,:i]
    else:
        vals,vecs=eigh(a,b,subset_by_index=[0, i-1])
        vals,vecs=vals.real,vecs.real
    return vals,vecs

def ch_index(data,clustering_labels):
    labels_unique=list(set(clustering_labels))
    N=len(data)
    k=len(labels_unique)
    cluster_centers=np.array(compute_centers(data,clustering_labels))
    set_center=np.array([sum(i) for i in zip(*data)])/N
    vols_dist=[len([i for i in clustering_labels if i==cluster])*np.linalg.norm(cluster_centers[j]-set_center) for j,cluster in enumerate(labels_unique)]
    intra_dist=[sum([np.linalg.norm(datum-cluster_centers[j]) for i,datum in enumerate(data) if clustering_labels[i]==j]) for j in labels_unique]
    ch=(N-k)/(k-1)*(sum(vols_dist))/sum(intra_dist)
    
    return ch

def compute_unsupervised_gsc(data,n_eig,graph,laplacian,max_it,n_clusters,use_minibatch):
   ch_list,vals_list,labels_list=[],[],[]

   for j in range(max_it):
       gsc_params=(2**j,1,1)
       vals,labels_unord=cluster(n_clusters,n_eig,graph.laplacian(laplacian,gsc_params),use_minibatch)
       vals_list.append(vals)
       labels_list.append(labels_unord)
       ch_list.append(ch_index(data,labels_unord))
   argmax=np.argmax(ch_list)
   print(argmax)
   return vals_list[argmax],labels_list[argmax]
       
def spectral_clustering(data,k_neighbors=None,n_eig=None,laplacian='rw',
                        g_method='knn',sym_method=None,sigma=1,gsc_params=None,unsupervised_gsc=False,
                        use_minibatch=True,eigen_only=False,clusters_fixed=False,return_matrix=False,
                        max_it=5,labels_given=np.array([None])):
    """Performs spectral clustering on a dataset of n-dimensional points.

    Inputs :
        data (ndarray): The dataset, a 
        k_neighbors (int): Number of neighbors you want to connect in the case of a k-nn graph.
        n_eig (int): Number of eigenvectors to calculate. Used to compute the number of clusters
        laplacian (string) : The laplacian to use between [un_norm , sym , rw] 
        sym_method (string): The method used to symmetrize the graph matrix in the case of an asymmetric adjacency matrix.
        sigma (float): Standard deviation for the gaussian kernel.
        gsc_params (3-uple): (t,alpha,gamma) for the gsc laplacians.
        use_minibatch (bool) : Choice of the k-means algorithm. True might lead to better performance on large datasets. Default = False.
        eigen_only (bool) : If True, the function will only returns the eigenvalues and eigenvectors and not compute the full clustering. Default = False.
        clusters_fixed (int) : The number of clusters in your data. If unknown, leave by default and the eigengap heuristic will be used. Default = False.
        return_matrix (bool) : True <=> returns the adjacency matrix alongside the clustering results. Use if you want to visualize the graph. Default = False.
        labels_given (ndarray) : The correct labels of your data. If given, used to reorder the labels obtained by clustering. Leave empty if unknown. Default = False 
    
    Returns :
        vals (ndarray): the computed eigenvalues pf the graph laplacian.
        labels (ndarray) : labels of the points after spectral clustering, ordered in the same way as the dataset.
        matrix (ndarray) : the adjacency matrix of the graph
        """
    
    #Choosing the optimal  base parameters
    N=len(data)
    if n_eig==None:
        n_eig=N
    if k_neighbors==None:
        k_neighbors=int(np.floor(np.log(N)))
    print("Building dataset graph...")
    graph=Graph(data,k_neighbors,g_method,sym_method,sigma)
    dir_status=['directed','undirected'][int(issymmetric(graph.m))]
    print(f"Dataset's {dir_status} graph built. ")
    
    print("Performing spectral embedding on the data.")
    if not unsupervised_gsc:
        if eigen_only:
            vals,u_full=eigenvectors(n_eig,*graph.laplacian(laplacian,gsc_params))
            return vals
        vals,labels_unord=cluster(clusters_fixed,n_eig,graph.laplacian(laplacian,gsc_params),use_minibatch)
    else:
        #Choosing the optimal gsc parameters and clustering.
        if laplacian=='rw':
            laplacian='g_rw'
        vals,labels_unord=compute_unsupervised_gsc(data,n_eig,graph,laplacian,max_it,clusters_fixed,use_minibatch)
    
    if np.all(labels_given)==None:
        #If the labels aren't given, we order labels based on cluster centroids.
        #We do not use the KMeans clusters_centers attribute because we 
        #need to compute them on the original dataset, not in the spectral embedding.
        centers=compute_centers(data,labels_unord)
        labels_ordered = reorder_labels(centers,labels_unord)
    else:
        #If the true labels are given, we order our clustering labels by inference.
        cluster_labels = infer_cluster_labels(clusters_fixed, labels_given,labels_unord)
        labels_ordered = infer_data_labels(labels_unord,cluster_labels)
        
    if return_matrix:
        return vals,labels_ordered,graph.m

    return vals,labels_ordered

