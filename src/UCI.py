'''Evaluation of SC methods on 11 UCI datasets. 
-100 iterations for each method
-Use of the optimal parameters given by GSC paper
-NMI score for evaluating performance
-Returns CSV ig


TODO : load control cjhart, breast tissue and seeds
fix labels to int for infer_labels.'''

from spectral_clustering import*
from sklearn import metrics
import ucimlrepo as uci
import pandas

['iris','glass','wine','wbdc','control chart','parkinson','vertebral','breast tissue','seeds','image seg','yeast']
def import_dataset(c:str):
    """Import one of the 11 UCI benchmark datasets :
        IRIS, GLASS, WINE, WBDC, CONTROL CHART; PARKINSON, VERTEBRAL,
        BREAST TISSUE, SEEDS, IMAGE SEG., YEAST"""
    
    set_table={'iris': 53, 'glass':42,'wine':109,'wbdc':336,'parkinson':174,'vertebral': 212,'image seg':50,'yeast':110}
    
    if set_table.get(c)!=None:
        dataset=uci.fetch_ucirepo(id=set_table[c])
        data_df=dataset.data.features
        labels_df=dataset.data.targets
        data=data_df.to_numpy()
        labels=labels_df.to_numpy()
    else:
        pass
    return data, labels


def nmi_clustering(c:str,options:dict):

    data,labels=import_dataset(c)
    vals,labels_spectral=spectral_clustering(data,options['k'],options['n_eig'],options['laplacian'],options['graph'],options['sym_method'],options['sigma'],labels_given=labels)
    nmi=metrics.normalized_mutual_info_score(labels,labels_spectral)
    
    return nmi


options={'k': 10, 'n_eig': 12, 'laplacian': 'rw', 'graph': 'knn', 'sym_method':'mean','sigma':1/2}
    
print(nmi_clustering('iris',options))