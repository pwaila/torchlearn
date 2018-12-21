###
import _pickle as cPickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import heapq
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

datapath = '/home/pune1/pranav/Deeplearning/MNISTData/'

# LOADING THE DICTIONARY FROM A PICKLE
with open(datapath+'MNISTData.pkl', 'rb') as fp :
    new_dict = cPickle.load(fp)


#print(new_dict['train_images'][0])

#print(len(new_dict['train_labels']))
#print(len(new_dict['test_labels']))

def l1_dist_formula(list1,list2):
    squared_sum=0
    for i in range(len(list1)):
        squared_sum+=(math.pow(list1[i]-list2[i],2))
    dist=math.sqrt(squared_sum)
    #print(dist)
    return dist

def l2_dist_formula(list1,list2):
    squared_sum=0
    for i in range(len(list1)):
        squared_sum+=(math.pow(list1[i]-list2[i],2))
    dist=math.sqrt(squared_sum)
    #print(dist)
    return dist

def l2_dist(list1,list2):
    #squared_sum=0
    '''
    for i in range(len(list1)):
        squared_sum+=(math.pow(list1[i]-list2[i],2))
    '''
    #sub_l=list1-list2
    #squared_sum=sum(math.pow(list1[i]-list2[i],2))
    #dista=math.sqrt(squared_sum)
    dista=distance.euclidean(list1,list2)
    #print(dista)
    return dista

#zero_mat= np.zeros((len(new_dict['test_labels']),len(new_dict['train_labels'])))
neighbor_size=3
prediction=[]
for i in range(len(new_dict['test_labels'])):
    flat_test = [item for sublist in new_dict['test_images'][i] for item in sublist]
    dist_list=[]
    for j in range(len(new_dict['train_labels'])):
        flat_train = [item for sublist in new_dict['train_images'][j] for item in sublist]
        #zero_mat[i,j]=l1_dist(flat_train,flat_tes)
        #print(zero_mat)
        dist_list.append(l2_dist(flat_train,flat_test))
    #nearest neighbours of given neighbor_size
    ax_index=heapq.nlargest(neighbor_size,range(len(dist_list)),np.asarray(dist_list).take)
    #finding the lables of nearest neighbors
    L=[new_dict['train_labels'][i] for i in ax_index]
    #finding the most common value among the nearest mabors
    most_common,num_most_common = Counter(L).most_common(1)[0]
    #appending the result in the prediction list
    prediction.append(most_common)
    #print("=======Printing the restults========")
    print(i)
    #print(dist_list)
    #print(ax_index)
    #predicted
    ##print("predicted")
    ##print(most_common)
    #Actual
    ##print("actual")
    ##print(new_dict['test_labels'][i])
    print("===============")
#accuracy score of the algo
print("accuracy score")
print(accuracy_score(new_dict['test_labels'],prediction))


'''
#print(len(zero_mat[0,:]))
#print(len(zero_mat[:,0]))
neighbour = 3
import heapq
result=[]
for i in range(zero_mat.shape[0]):
    #for j in range(len(z[0])):
    #print(z[i])
    max_index=heapq.nlargest(neighbour,range(len(zero_mat[i])),np.asarray(zero_mat[i]).take)
    L=[new_dict['train_labels'][i] for i in max_index]
    from collections import Counter
    most_common,num_most_common = Counter(L).most_common(1)[0]
    result[most_common]
print("final result is :")    
print(result)
'''