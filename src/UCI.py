'''Evaluation of SC methods on 11 UCI datasets. 
-100 iterations for each method
-Use of the optimal parameters given by GSC paper
-NMI score for evaluating performance


TODO : weird convergence errors for (wbdc,all),(parkinson,all) #PROBABLY INFER CLUSTER LABELS AT FAULT.
       '''

from spectral_clustering import*
from sklearn import metrics
from sklearn import datasets
from json import loads
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


def nmi_clustering(c:str,options:dict,laplacian):
    n_clusters,data,labels=import_dataset(c)
    labels_spectral,vals=spectral_clustering(data,n_clusters=n_clusters,k_neighbors=options[c],laplacian=laplacian)
    nmi=metrics.normalized_mutual_info_score(labels,labels_spectral)
    
    return nmi

def cluster_average(c:str,options:dict,avg:int,laplacian):
    nmi=0
    for i in range(avg):
        nmi+=nmi_clustering(c,options,laplacian)
    return nmi/avg

def benchmark(avg:int,options_dict:dict[dict],laplacian:str):
    averages=dict()
    #['iris' ,'glass','wine','wbdc','parkinson','vertebral','breast tissue','seeds','image seg','yeast']
    for dataset in ['iris' ,'glass']:
        print(dataset)
        averages[dataset]=cluster_average(dataset,options_dict,avg,laplacian)*100
    return averages

def full_benchmark(avg:int,options_dict:dict[dict],laplacians:list):
    results=dict()
    for laplacian in laplacians:
        results[laplacian]=benchmark(avg,options_dict,laplacian)
    return results

avg=1

"""FULL BENCHMARK :"""
k_dict={'iris':4,
              'glass':6,
               'wine': 3,
               'wbdc':2,
               'control chart':6,
               'parkinson':2,
               'vertebral':3,
               'breast tissue':6,
               'seeds':3,
               'image seg':7,
               'yeast':10}
laplacians=['un_norm','sym','rw','g','g_rw']
with open('uci_results.txt','w') as f:
    f.write(str(full_benchmark(avg,k_dict,laplacians)))


