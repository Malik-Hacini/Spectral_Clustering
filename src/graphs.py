import numpy as np
import time

def full_kernel(N,nodes,similarity):
     matrix=np.zeros((N,N))
     for i in range(N):
          for j in range(i+1):
               if j!=i: #No node is connected to itself (no impact on clustering)
                matrix[i,j]=similarity.gaussian(nodes[i],nodes[j])
                matrix[j,i]=matrix[i,j]
    
     return matrix + matrix.T


def knn(k: int, nodes, similarity):
    """Constructs a k-nearest neighbor graph of the given nodes (n-dimensional points) using the similarity function given.
    
        Inputs :
            k (int): Number of neighbors you want to connect.
            nodes (list) : Nodes of the graph as n-dimensional points.
            similarity (Similarity) : Object of the Similarity class (contains multiples similarity functions)
        
        Returns :
            matrix (ndarray) : k-nearest neighbors weighted similarity matrix."""
    N = len(nodes)
    matrix = np.zeros((N, N))

    start=time.time()
    distances=np.zeros((N,N))
    # Calculate distances
    for i in range(N):
        for j in range(i+1, N):
            distances[i, j] = similarity.gaussian(nodes[i], nodes[j])
        print(f"Node {i+1}/{N} distances computed. ")
    distances=distances+distances.T

    print(f"distances computed in {time.time()-start} s. ")
    
    # Find k-nearest neighbors
    for i in range(N):
        knn_indices = np.argpartition(distances[i], -k)[-k:]
        for j in knn_indices:
            if j != i:
                matrix[i, j] = 1
        print(f"Node {i+1}/{N} neighbors computed.")

    return matrix


def knn_gaussian(k: int, N, nodes, similarity):
        """Constructs a k-nearest neighbor graph of the given nodes (n-dimensional points) using the gaussian similarity function.
        The graph is weighted : W(i,j)=0 if j isn't in the k-nearest neighbors of i and W(i,j)=gaussian(i,j) else.
    Inputs :
        k (int): Number of neighbors you want to connect.
        N (int): Number of nodes
        nodes (list) : Nodes of the graph as n-dimensional points.
        similarity (Similarity) : Object of the Similarity class (contains multiples similarity functions)
    
    Returns :
        matrix (ndarray) : k-nearest neighbors weighted similarity matrix."""
    
        matrix=np.zeros((N,N))
        for i in range(N):
            neighbors_similarities=[]
            for neighbor in nodes:
                #Change similariy function here after adding it to the similarity class.
                neighbors_similarities.append(similarity.gaussian(nodes[i],neighbor))
            knn_ind=np.argpartition(neighbors_similarities,-k)[-k:]   
            for ind in knn_ind:
                if ind!=i: #No node is connected to itself (no impact on clustering)
                    matrix[i, ind]= neighbors_similarities[ind]
        return matrix

def symmetrize(matrix,method):
        """Symmetrizes knn matrix using different methods.
        Inputs :
            matrix (ndarray): the matrix to symmetrize (must be real valued)
            method (string): the method used, must be in ['mean','and','or']
        Returns :
            matrix (ndarray) : The symmetrized matrix."""
        
        if method=='mean':
            return (1/2)*(matrix+matrix.T)
        
        "PROBLEM ON AND ???"
        if method=='and':
            for i in range(matrix.shape[0]):
                 for j in range(i+1):
                      if matrix[i,j]!=matrix[j,i]:
                           matrix[i,j]=0
                           matrix[j,i]=0

        if method=='or':
             for i in range(matrix.shape[0]):
                 for j in range(i+1):
                      if matrix[i,j]!=matrix[j,i]:
                           if matrix[i,j]!=0:
                                matrix[j,i]=matrix[i,j]
                           elif matrix[j,i]!=0:
                                matrix[i,j]=matrix[j,i]

        return matrix


class Similarity:
    """Used to store different similarity functions between n-dimensional points."""
    def __init__(self,sigma=None):
        """Sigma is a standard deviation parameter used for certain fucntions."""
        self.sigma=sigma

    def euclidean(self,x1 : list,x2: list,*args)->float:
        '''Calculates the euclidean distance between two n-dimensional points x1 and x2'''
        return np.linalg.norm(np.subtract(x1,x2))

    def gaussian(self,x1: list,x2: list)->float:
        '''Returns the gaussian kernel of standard deviation sigma between two n-dimensional points x1 and x2.'''
        return np.exp(-(self.euclidean(x1,x2)**2)/(2*(self.sigma**2)))
    

class Graph:
    """Similarity graphs based on a dataset of n-dimensional vectors"""
    def __init__(self,data,k,g_method,sym_method,sigma=None) -> None:
        """
    Inputs :
        data (ndarray): the dataset as an array of n-dimensional points.
        k (int): Number of neighbors you want to connect.
        sigma (float): standard deviation (only if you use a similarity function with this parameter)
     """
    
        self.dim=len(data[0])
        self.N=len(data)
        self.nodes=data
        self.similarity=Similarity(sigma)
        if g_method=='knn':
             self.m=symmetrize(knn(k,self.nodes,self.similarity),sym_method)
        if g_method=='g_knn':
            self.m=symmetrize(knn_gaussian(k,self.N,self.nodes,self.similarity),sym_method)
        if g_method=='f_kernel':
            #Lower sigma needed for good performance.
            self.m=full_kernel(self.N,self.nodes,self.similarity)
        self.degree_m=np.diag([sum(self.m[i]) for i in range(self.N)])


    def laplacian(self):
        """Constructs the graph laplacian D - W based on the graph matrix W."""
        return np.subtract(self.degree_m, self.m)
    

    def laplacian_sym(self):
        """Constructs the graph laplacian I - D^(-1/2)WD^(-1/2) based on the graph matrix W."""
        inv_sqrt_d=np.diag([1/np.sqrt(sum(self.m[i])) for i in range(self.N)])
        return np.subtract(np.identity(self.N),np.matmul(inv_sqrt_d,np.matmul(self.m,inv_sqrt_d)))

    def laplacian_rw(self):
        """Constructs the graph laplacian I - D^(-1)W based on the graph matrix W. """
        inv_d=np.diag([1/sum(self.m[i]) for i in range(self.N)])
        o=np.matmul(inv_d,self.m)
        return np.subtract(np.identity(self.N),np.matmul(inv_d,self.m))
