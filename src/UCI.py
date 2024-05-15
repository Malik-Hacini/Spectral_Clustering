'''Evaluation of SC methods on 11 UCI datasets. 
-100 iterations for each method
-Use of the optimal parameters given by GSC paper
-NMI score for evaluating performance


TODO : weird convergence errors for (wine,all), (wbdc,all),(parkinson,all)
       '''

from spectral_clustering import*
from sklearn import metrics
from sklearn import datasets
import ucimlrepo as uci
import pandas

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
            
            data=np.loadtxt(f'src/Datasets/{c}_dataset.txt')
            labels=np.loadtxt(f'src/Datasets/{c}_labels.txt')

        n_clusters=len(set(labels))
    return n_clusters,data, labels


def nmi_clustering(c:str,options:dict):

    n_clusters,data,labels=import_dataset(c)
    vals,labels_spectral=spectral_clustering(data,options['k'],options['n_eig'],options['laplacian'],options['graph'],options['sym_method'],options['sigma'],labels_given=labels,clusters_fixed=n_clusters,use_minibatch=True)
    nmi=metrics.normalized_mutual_info_score(labels,labels_spectral)
    
    return nmi

def cluster_average(c:str,options:dict,avg:int):
    nmi=0
    for i in range(avg):
        nmi+=nmi_clustering(c,options)
    return nmi/avg

options={'k': 10, 'n_eig': 12, 'laplacian': 'rw', 'graph': 'knn', 'sym_method':'mean','sigma':1/2}
def benchmark(avg:int,options_dict:dict[dict],laplacian:str):
    averages=dict()
    for dataset in ['iris' ,'glass','wine','wbdc','control chart','parkinson','vertebral','breast tissue','seeds','image seg','yeast']:
        print(dataset)
        options_dict[dataset]['laplacian']=laplacian
        averages[dataset]=cluster_average(dataset,options_dict[dataset],avg)*100
    return averages

def full_benchmark(avg:int,options_dict:dict[dict],laplacians:list):
    results=dict()
    for laplacian in laplacians:
        results[laplacian]=benchmark(avg,options_dict,laplacian)
    return results
g_method='knn'
sym_method='mean'
sigma=1/2
avg=1

"""FULL BENCHMARK :
options_dict={'iris':{'k': 4, 'n_eig': 7, 'graph': g_method, 'sym_method':sym_method,'sigma':sigma},


              'glass':{'k': 6, 'n_eig': 8, 'graph': g_method, 'sym_method':sym_method,'sigma':sigma},
               'wine': {'k': 3, 'n_eig': 4, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'wbdc':{'k': 2, 'n_eig': 4, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'control chart':{'k': 6, 'n_eig': 8, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'parkinson':{'k': 2, 'n_eig': 3, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'vertebral':{'k': 3, 'n_eig': 4, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'breast tissue':{'k': 6, 'n_eig': 8, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'seeds':{'k': 3, 'n_eig': 5, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'image seg':{'k': 7, 'n_eig': 9, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma},
               'yeast':{'k': 10, 'n_eig': 4, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma}}
laplacians=['un_norm','sym','rw']
with open('uci_results.txt','w') as f:
    f.write(str(full_benchmark(avg,options_dict,laplacians)))"""


"""WBCDoptions={'k': 2, 'n_eig': 4, 'graph': g_method , 'sym_method':sym_method,'sigma':sigma, 'laplacian':'sym'}
n_clusters,data,labels=import_dataset('wbdc')
print(labels)
print(n_clusters)
vals,labels_spectral=spectral_clustering(data,options['k'],options['n_eig'],options['laplacian'],options['graph'],options['sym_method'],options['sigma'],labels_given=labels,use_minibatch=True,clusters_fixed=n_clusters)
print(labels_spectral)
print(metrics.normalized_mutual_info_score(labels,labels_spectral))"""