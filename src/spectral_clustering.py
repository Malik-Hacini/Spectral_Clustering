from graphs import*
from misc import*
from scipy.linalg import eigh
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



def k__eigenvectors(matrix,i):
    """Computes the first i eigenvals and eigenvecs of a sparse matrix.
    Inputs :
        matrix (ndarray): The matrix. Needs to be sparse.
        i (int): Number of eigenvals and eigenvecs to compute (in ascending order)
    
    Returns :
        vecs (ndarray) : Matrix with the i eigenvecs as columns."""
    
    vals,vecs=eigh(matrix,subset_by_index=[0, i-1])
    vals=vals.real
    vecs=vecs.real
    return vals,vecs

def eigengap(vals):
    X=np.arange(1,len(vals)+1,1)
    Y=vals
    plt.scatter(X,Y)
    plt.show()
    c=input("Number of clusters to construct:\n")
    
    return int(c)


def spectral_clustering(data,k_neighbors,n_eig,laplacian,g_method='g_knn',sym_method='mean',sigma=None,use_minibatch=False,eigen_only=False,clusters_fixed=False,return_matrix=False,labels_given=np.array([None])):
    """Performs spectral clustering on a dataset of n-dimensional points.
    Inputs :
        data (ndarray): dataset.
        k_neighbors (int): Number of neighbors you want to connect for the k-nn graph.
        n_eig (int): Number of eigenvectors to calculate. Used to compute the number of clusters
        laplacian (string) : The laplacian to use between [D - W , I - D^(-1/2)WD^(-1/2) , I - D^(-1)W] 
        sym_method (string): The method used to symmetrize the graph matrix if knn is used.
        sigma (float): Standard deviation (only if you use a similarity function with this parameter)
    
    Returns :
        vals (ndarray): the computed eigenvalues.
        labels (ndarray) : labels of the points after spectral clustering, ordered in the same way as the dataset."""
    print("Building dataset graph...")
    graph=Graph(data,k_neighbors,g_method,sym_method,sigma)
    print("Dataset graph built.")

    print("Performing spectral embedding on the data.")
    if laplacian=='un_norm':
        vals,u_full=k__eigenvectors(graph.laplacian(),n_eig)
    if laplacian=='sym':
        vals,u_full=k__eigenvectors(graph.laplacian_sym(),n_eig)
        u_full=np.apply_along_axis(normalize_vec, axis=0, arr=u_full)
    if laplacian=='rw':
        vals,u_full=k__eigenvectors(graph.laplacian_rw(),n_eig)

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

