from utils.GMM import*
from spectral_clustering import*
from utils.Plots.matplot_funcs import*
from utils.data_files_managing import*
from sklearn import datasets,metrics



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

def gsc_graph_eigen(data,labels,l,n_clusters,name,gsc_params=None,NMI=False):

    if l in ['g','g_rw']:
        sym_method=None
        directed=True
    else:
        sym_method='mean'
        directed=False
    vals,labels_spectral,matrix=spectral_clustering(data,4,n_clusters+1,l,'knn',sym_method=sym_method,gsc_params=gsc_params,
                                                    sigma=1,clusters_fixed=n_clusters,return_matrix=True,use_minibatch=True,
                                                    )
    if NMI:
        nmi_score=round(metrics.normalized_mutual_info_score(labels,labels_spectral)*100,4)
    else:
        nmi_score=None
    print("Plotting results...")
    plt=plot_sc_graph_eigengap(data,labels,labels_spectral,matrix,vals,l,directed=directed,nmi_score=nmi_score)
    save_plot(plt,f'{name}_{l}',dataset_name=name)



'''N=500
means=[(1,0),(1/2,1/2),(1,1)]
distrib=[0.1,0.35,0.55]
sigmas_x=[1/6,1/6,1/6]
sigmas_y=[1/6,1/6,1/6]
p_list=[0.1,0.1,0.1]
covs=[bivariate_cov_m(sigmas_x[i],sigmas_y[i],p_list[i]) for i in range(len(means))]
#data,labels=GMM(2,N,means,covs,distrib)'''


gsc_params=(3,0.7,0.9)
data,labels,name=load_data_n_labels('circles')
for l in ['un_norm','sym','rw','g','g_rw']:
    gsc_graph_eigen(data,labels,l,n_clusters=3,name=name,gsc_params=gsc_params,NMI=True)

save_data_n_labels(data,labels,'test1')
