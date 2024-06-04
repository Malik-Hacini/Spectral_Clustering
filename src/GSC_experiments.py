from utils.GMM import*
from spectral_clustering import*
from utils.Plots.matplot_funcs import*
from utils.data_files_managing import*
from sklearn import datasets,metrics

'''NOTE : everything here is broken, as the way the calls to spectral_clustering works have changed. '''


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
    '''N=100
    means=[(0,0),(1.15,1.15)]
    n_clusters=len(means)
    distrib=[0.5,0.5]
    sigmas_x=[1/3,1/3]
    sigmas_y=[1/3,1/3]
    p_list=[0,0]
    covs=[bivariate_cov_m(sigmas_x[i],sigmas_y[i],p_list[i]) for i in range(len(means))]
    gsc_params=(3,0.7,0.9)

    data,labels=GMM(n_clusters,N,means,covs,distrib)     

    #data,labels,name=load_data_n_labels('low_high_asym2_blobs')
    name=f"circ_{N}_sym"
    for l in ['rw','g_rw']:
        gsc_graph_eigen(data,labels,l,n_clusters=2,name=name,gsc_params=gsc_params,NMI=True)

    save_data_n_labels(data,labels,name)'''
    if l in ['g','g_rw']:
        directed=True
    else:
        directed=False
    labels_spectral,vals,matrix=spectral_clustering(data,n_clusters=n_clusters,k_neighbors=4,n_eig=n_clusters+1,laplacian=l,gsc_params=gsc_params,
                                                    sigma=1,return_matrix=True,use_minibatch=True,
                                                    )
    if NMI:
        nmi_score=round(metrics.normalized_mutual_info_score(labels,labels_spectral)*100,4)
    else:
        nmi_score=None
    print("Plotting results...")
    plt=plot_sc_graph_eigengap(data,labels,labels_spectral,matrix,vals,l,directed=directed,nmi_score=nmi_score)
    save_plot(plt,f'{name}_{l}',dataset_name=name)

def eigengap_total_curve(steps,gsc_params=None):
    '''gsc_params=(3,0.7,0.9)
    steps=[100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000]

    eigengap_total_curve(steps,gsc_params)'''
    means=[(0,0),(1.15,1.15)]
    n_clusters=len(means)
    distrib=[0.5,0.5]
    sigmas_x=[1/3,1/3]
    sigmas_y=[1/3,1/3]
    p_list=[0,0]
    covs=[bivariate_cov_m(sigmas_x[i],sigmas_y[i],p_list[i]) for i in range(len(means))]

    values={'rw': [],'g_rw': []}
    gaps={"rw":[],"g_rw":[]}
    for l in ('rw','g_rw'):
        for step in steps:
            print(l,step)
            data,labels=GMM(n_clusters,step,means,covs,distrib)            
            labels_spectral,vals=spectral_clustering(data,k_neighbors=6,n_eig=n_clusters+1,laplacian=l,gsc_params=gsc_params,
                                                    sigma=1,n_clusters=n_clusters,
                                                    )
            values[l].append(vals)
            gap=(vals[-1]-vals[-2])/vals[-2]
            gaps[l].append(gap)
    plt=plot_gapcurve(steps,gaps,[r'$L_{rw}$',r'$L_{G_{rw}}$'])
    with open ('eigencurve.txt','w') as f:
        f.write(str(values))
    
    save_plot(plt,'circ_eigencurve')

def test_unsupervised_gsc(l,name):
    data,labels,namee=load_data_n_labels(name)
    labels_spectral,vals,matrix=spectral_clustering(data,n_clusters=2,laplacian='g_rw',return_matrix=True)
    plt=plot_sc_graph_eigengap(data,labels,labels_spectral,matrix,vals,l,directed=True)
    save_plot(plt,f'{name}_{l}',dataset_name=name)


'''
means=[(0,0),(1.15,1.15)]
n_clusters=len(means)
distrib=[0.5,0.5]
sigmas_x=[1/3,1/3]
sigmas_y=[1/3,1/3]
p_list=[0,0]
covs=[bivariate_cov_m(sigmas_x[i],sigmas_y[i],p_list[i]) for i in range(len(means))]
data,labels=GMM(2,100,means,covs,distrib)'''
gsc_params=(1,1,0.99)
data,labels,name=load_data_n_labels('gsc_test')


labels_spectral,vals=spectral_clustering(data,k_neighbors=6,n_eig=3,laplacian='g_rw',gsc_params=gsc_params,
                                                    sigma=1,n_clusters=2
                                                    )
