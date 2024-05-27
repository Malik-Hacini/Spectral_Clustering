import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def save_plot(plt,name,path='Results/Saved_plots',show=True):
    fig = plt.gcf()
    fig.set_size_inches(14.40, 8.10)
    plt.subplots_adjust(left=0.005,bottom=0.323,right=0.995,top=0.683,hspace=0.2,wspace = 0)
    plt.savefig(f"{path}/{name}.svg",dpi=100)
    print("Plot saved successfully.")
    if show:
        plt.show()

def plot_clustering(data,labels):
    """Plots the clustering of 2D data based on the given labels.
    
    Inputs :
        data (ndarray): ndarray of the data. Shape : [[x1,y1],[x2,y2]...]
        labels (ndarray) : ndarray of the labels (integers). Shape : [0,1,2...]"""
    x,y=data.T
    plt.scatter(x,y,c=labels.astype(float))
    plt.show()

def plot_clustering_vs_gt(data,labels_clustering,labels):
    """Plots the clustering of 2D data against the ground truth of the data.
    
    Inputs :
        data (ndarray): ndarray of the data. Shape : [[x1,y1],[x2,y2]...]
        labels_clustering (ndarray) : ndarray of the labels obtained by clustering (integers). Shape : [0,1,2...]
        labels (ndarray): ndarray of the ground truth labels.
        titles (list): titles of the subplot for each method."""
    x,y=data.T
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clustering performance testing')
    ax1.title.set_text('Clustering')
    ax2.title.set_text('Ground Truth')
    ax1.scatter(x,y,c=labels_clustering.astype(float))
    ax2.scatter(x,y,c=labels.astype(float))
    plt.show()

def plot_all_clustering_vs_gt(data,labels,titles):
    """Plots the clustering of 2D data against the ground truth of the data, with all the given methods.
    
    Inputs :
        data (ndarray): ndarray of the data. Shape : [[x1,y1],[x2,y2]...]
        labels (ndarray) : list of ndarrays of the labels to plot.
        titles (list): titles of each plot"""
    
    x,y=data.T
    n_sub=len(labels)
    n_1=int(np.floor((n_sub+1)/2))
    #fig, axs=plt.subplots(n_sub - n_1,n_1)
    fig, axs=plt.subplots(2,2)
    fig.suptitle('Comparison of spectral clustering methods')
    for i,ax in enumerate(axs.flat):
        ax.scatter(x,y,c=labels[i].astype(float))
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    

    plt.show()

def plot_eigenvalues(values,titles):
    X=np.arange(1,len(values[0])+1,1)
    fig, axs=plt.subplots(nrows=1,ncols=len(values))
    for i,ax in enumerate(axs.flat):
        ax.locator_params(axis="x", integer=True, tight=True)
        ax.locator_params(axis="y", tight=True,nbins=4)
        ax.scatter(X,values[i])
        
    plt.show()

def boxplot(Y,title):
    plt.suptitle(title)
    plt.boxplot(Y)
    plt.show()

def multiboxplot(settings_title,Y,titles=False):
    n_rows=len(Y)
    if not titles:
        titles=[" " for i in range(n_rows)]
    fig, axs=plt.subplots(nrows=n_rows)
    fig.suptitle(settings_title)

    for i,ax in enumerate(axs):
        ax.boxplot(Y[i])
        #plt.xticks(ticks=np.arange(0,4,1),labels=titles,fontsize=10)
        ax.set_title(titles[i])
    plt.show()


def precision_single_plot(data,labels_spectral,labels):
    x,y=data.T
    precision=round(len([label for i,label in enumerate(labels) if labels_spectral[i]==label])/len(labels),4)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Clustering performance : {precision*100}%')
    ax1.title.set_text('Clustering')
    ax2.title.set_text('Ground Truth')
    ax1.scatter(x,y,c=labels_spectral.astype(float))
    ax2.scatter(x,y,c=labels.astype(float))
    plt.show()


def plot_sc_graph(data,labels,labels_spectral,matrix):
    options = {"with_labels":False,"edgecolors": "tab:gray", "node_size": 50, "width": 0.5,"alpha": 1}
    x,y=data.T
    #precision=round(len([label for i,label in enumerate(labels) if labels_spectral[i]==label])/len(labels),4)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle(f'Clustering performance : {precision*100}%')
    ax1.title.set_text('Spectral clustering')
    ax2.title.set_text('Ground Truth')
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    G=nx.from_numpy_array(matrix)
    nodes=[i for i in range(len(data))]
    pos={i: datum for i,datum in enumerate(data)}

    
    nx.draw_networkx(G,pos,node_color=labels_spectral,**options,ax=ax1)
    ax2.scatter(x,y,c=labels.astype(float),edgecolors='gray')
    plt.show()


def plot_fig1(data,labels_clustering,labels_kmeans):
    """Plots the clustering of 2D data against the ground truth of the data.
    
    Inputs :
        data (ndarray): ndarray of the data. Shape : [[x1,y1],[x2,y2]...]
        labels_clustering (ndarray) : ndarray of the labels obtained by clustering (integers). Shape : [0,1,2...]
        labels (ndarray): ndarray of the ground truth labels.
        titles (list): titles of the subplot for each method."""
    x,y=data.T
    fig, (ax1, ax2) = plt.subplots(1, 2)
    colors=['b','r']
    labels_clustering=[colors[label] for label in list(labels_clustering)]
    labels_kmeans=[colors[label] for label in list(labels_kmeans)]

    ax2.title.set_text('Spectral clustering')
    ax1.title.set_text('k-means clustering')
    ax2.scatter(x,y,c=labels_clustering)
    ax1.scatter(x,y,c=labels_kmeans)
    plt.show()

def plot_fig3_binorm(data,labels):
    """Plots the clustering of 2D data based on the given labels.
    
    Inputs :
        data (ndarray): ndarray of the data. Shape : [[x1,y1],[x2,y2]...]
        labels (ndarray) : ndarray of the labels (integers). Shape : [0,1,2...]"""
    x,y=data.T
    labels=np.array([i+1 for i in labels])
    scatter=plt.scatter(x,y,c=labels)
    legend1 = plt.legend(*scatter.legend_elements(),
                    loc="upper left", title="Gaussians")
    plt.show()

def plot_sc_graph_eigengap(data,labels,labels_spectral,matrix,vals,l,directed=False,nmi_score=None):
    options = {"with_labels":False,"edgecolors": "tab:gray", "node_size": 50, "width": 0.5,"alpha": 1}
    if directed:
        options['arrowstyle']='-|>'
        options['arrowsize']=10
    x,y=data.T
    fig, (ax1, ax2, ax3)= plt.subplots(nrows=1,ncols=3,width_ratios=[1,1/2,1/3])
    if nmi_score==None:
        fig.suptitle(f'Laplacian : {l}.')
    else:
        fig.suptitle(f'Laplacian : {l}. NMI : {nmi_score}')
    ax1.title.set_text('Spectral clustering')
    ax2.title.set_text('Ground Truth')
    ax3.title.set_text('First eigenvalues')
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if directed:
        G=nx.from_numpy_array(matrix,create_using=nx.DiGraph)
    else:
        G=nx.from_numpy_array(matrix)
    pos={i: datum for i,datum in enumerate(data)}
    
    nx.draw_networkx(G,pos,node_color=labels_spectral,arrows=directed,**options,ax=ax1)
    ax2.scatter(x,y,c=labels.astype(float),edgecolors='gray')

    X=np.arange(1,len(vals)+1,1)
    ax3.locator_params(axis="x", integer=True)
    ax3.get_yaxis().set_visible(False)
    ax3.scatter(X,vals)

    
    return plt

