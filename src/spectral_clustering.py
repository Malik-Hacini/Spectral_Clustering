from graphs import*
from misc import*
from scipy.linalg import eigh,eig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans




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



def eigenvectors(i,a,b=None,):
    """Computes the first i eigenvals and eigenvecs of a symmetric matrix for the generalized problem
    a*X=lambda*b*X
    Inputs :
        a,(ndarray): Needs to be real symmetric.
        i (int): Number of eigenvals and eigenvecs to compute (in ascending order)
        b (ndarray) : The second matrix
    Returns :
        vecs (ndarray) : Matrix with the i eigenvecs as columns."""
    if b=='no':
        vals,vecs=eig(a)
    vals,vecs=eigh(a,b,subset_by_index=[0, i-1])
    vals=vals.real
    vecs=vecs.real
    return vals,vecs

def eigengap(vals):
    X=np.arange(1,len(vals)+1,1)
    Y=vals
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.locator_params(axis="y", tight=True,nbins=4)
    plt.scatter(X,Y)
    plt.show()
    c=input("Number of clusters to construct:\n")
    
    return int(c)


def spectral_clustering(data,k_neighbors,n_eig,laplacian,g_method='g_knn',sym_method=None,sigma=None,gsc_params=None,use_minibatch=False,eigen_only=False,clusters_fixed=False,return_matrix=False,labels_given=np.array([None])):
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
    print("Building dataset graph...")
    graph=Graph(data,k_neighbors,g_method,sym_method,sigma)
    print("Dataset graph built.")

    print("Performing spectral embedding on the data.")
    vals,u_full=eigenvectors(n_eig,*graph.laplacian(laplacian,gsc_params))

    #In the case of L_sym being used, there is an additional normalization step.
    if laplacian=='sym':
        u_full=np.apply_along_axis(normalize_vec, axis=0, arr=u_full)

    #Use of the eigengap heuristic to get the number of clusters, if it is not given
    if eigen_only:
        return vals
    
    if not clusters_fixed:
        n_clusters=eigengap(vals)
    else:
        n_clusters=clusters_fixed
    
    u=u_full[:,:n_clusters]

    print("k-means clustering the spectral embedded data...")
    labels_unord=kmeans(u,n_clusters,use_minibatch)
    clusters=[[datum for j,datum in enumerate(data) if labels_unord[j]==i] for i in range(n_clusters)]
    if np.all(labels_given)==None:
        #If the labels aren't given, we order labels based on cluster centroids.
        #We do not use the KMeans clusters_centers attribute because we 
        #need to compute them on the original dataset, not in the spectral embedding.
        centers=[]
        for i in range(n_clusters):
            avg=[]
            for j in range(2):
                avg.append(sum([t[j] for t in clusters[i]])/len(clusters[i]))
            centers.append(avg)
        labels_ordered = reorder_labels(centers,labels_unord)
    else:
        #If the true labels are given, we order our clustering labels by inference.
        cluster_labels = infer_cluster_labels(n_clusters, labels_given,labels_unord)
        labels_ordered = infer_data_labels(labels_unord,cluster_labels)

        
    if return_matrix:
        return vals,labels_ordered,graph.m

    return vals,labels_ordered

