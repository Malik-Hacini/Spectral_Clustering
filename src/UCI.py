'''Evaluation of SC methods on 11 UCI datasets. 
-100 iterations for each method
-Use of the optimal parameters given by GSC paper
-NMI score for evaluating performance


TODO : weird convergence errors for (wbdc,all),(parkinson,all) #PROBABLY INFER CLUSTER LABELS AT FAULT ? idk
       '''

from spectral_clustering import*
from sklearn import metrics
from sklearn import datasets
from json import loads
from utils.Plots.matplot_funcs import*
import ucimlrepo as uci
import pandas


def download_everything():
    datas=dict()
    for set in ['iris' ,'glass','wine','wbdc','parkinson','vertebral','breast tissue','seeds','image seg','yeast']:
        datas[set]=import_dataset(set)
        print(set)
    with open('uci_data.txt','w') as f:
        f.write(str(datas))

def load_everything():
    with open('src/utils/Datasets/uci_data.txt','r') as f:
        data=f.read()
    data_dict=loads(data)
    return data_dict


#n_clusters={'iris': 3 ,'glass': 6,'wine':3,'wbdc':2,'control chart':6,'parkinson':2,'vertebral':3,'breast tissue':6,'seeds':3,'image seg':7,'yeast':3}
def import_dataset(c:str):
    """Import one of the 11 UCI benchmark datasets :
        IRIS, GLASS, WINE, WBDC, CONTROL CHART; PARKINSON, VERTEBRAL,
        BREAST TISSUE, SEEDS, IMAGE SEG., YEAST"""
    
    set_table={'iris': 53, 'glass':42,'wine':109,'parkinson':174,'vertebral': 212,'image seg':50,'yeast':110}
    
    
    if set_table.get(c)!=None:
        dataset=uci.fetch_ucirepo(id=set_table[c])
        data_df=dataset.data.features
        labels_df=dataset.data.targets
        data=data_df.to_numpy()
        labels=labels_df.to_numpy()
        labels=labels.flatten()
        n_clusters,labels=labels_to_ints(labels)
    else:
        if c=='wbdc':
            data,labels=datasets.load_breast_cancer(return_X_y=True)
        else:
            if c=='breast tissue':
                c='breast_tissue'
            if c=='control chart':
                c='control_chart'
            
            data=np.loadtxt(f'src/utils/Datasets/{c}_dataset.txt')
            labels=np.loadtxt(f'src/utils/Datasets/{c}_labels.txt')

        n_clusters=len(set(labels))
    return n_clusters,data, labels


def nmi_clustering(c:str,k_dict:dict,laplacian,gsc_dict:dict):
    n_clusters,data,labels=import_dataset(c)
    labels_spectral,vals=spectral_clustering(data,n_clusters=n_clusters,n_eig=n_clusters+1,k_neighbors=k_dict[c],laplacian=laplacian,gsc_params=gsc_dict[c])
    nmi=metrics.normalized_mutual_info_score(labels,labels_spectral)
    return nmi,vals



def benchmark(options_dict:dict[dict],laplacian:str,gsc_dict,rep):
    averages=dict()
    values=[]
    titles=[laplacian]+['iris' ,'glass','wine','wbdc','parkinson','vertebral','breast tissue','seeds','image seg','yeast']
    #['iris' ,'glass','wine','wbdc','parkinson','vertebral','breast tissue','seeds','image seg','yeast']
    for i in range(rep):
        print(laplacian,i)
        for dataset in ['iris' ,'glass','wine','wbdc','parkinson','vertebral','breast tissue','seeds','image seg','yeast']:
            print(dataset)
            nmi,vals=nmi_clustering(dataset,options_dict,laplacian,gsc_dict)
            if i==0:
                averages[dataset]=nmi*100
            if nmi*100>averages[dataset]:
                averages[dataset]=nmi*100
            values.append(list(vals))

    #plt=plot_eigenvalues(values,titles,uci=True)
    #save_plot(plt,f'UCI_{laplacian}',show=True,adjust=False)
    
    return averages

def full_benchmark(options_dict:dict[dict],laplacians:list,gsc_dict,rep):
    results=dict()
    for laplacian in laplacians:
        results[laplacian]=benchmark(options_dict,laplacian,gsc_dict,rep)
    return results



"""FULL BENCHMARK :"""
k_dict={'iris':4,
              'glass':6,
               'wine': 4,
               'wbdc':5,
               'control chart':6,
               'parkinson':4,
               'vertebral':3,
               'breast tissue':6,
               'seeds':3,
               'image seg':7,
               'yeast':10}

gsc_dict={'iris':(32,0.2,0.99),
              'glass':(32,0.1,0.99),
               'wine': (32,0.1,0.99),
               'wbdc':(64,0.1,0.99),
               'control chart':(16,0.4,0.99),
               'parkinson':(64,0.1,0.99),
               'vertebral':(32,0.2,0.99),
               'breast tissue':(16,0.1,0.99),
               'seeds':(2,0.8,0.99),
               'image seg':(64,0.1,0.99),
               'yeast':(2,0.2,0.99)}
laplacians=['un_norm','sym','rw','g','g_rw']
with open('uci_results.txt','w') as f:
    f.write(str(full_benchmark(k_dict,laplacians,gsc_dict,10)))


