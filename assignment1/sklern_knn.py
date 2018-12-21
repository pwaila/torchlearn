###
import _pickle as cPickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import heapq
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

datapath = '/home/pune1/pranav/Deeplearning/MNISTData/'

# LOADING THE DICTIONARY FROM A PICKLE
with open(datapath+'MNISTData.pkl', 'rb') as fp :
    new_dict = cPickle.load(fp)

def flatten(lol):
    flat=[item for sublist in lol for item in sublist]
    return flat

train_X= [flatten(new_dict['train_images'][i]) for i in new_dict['train_labels']] 
train_y= new_dict['train_labels']

validate_X= [flatten(new_dict['test_images'][i]) for i in new_dict['test_labels']] 
validate_y= new_dict['test_labels']

#train_data = train_df.values

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=0)


knn_clf=KNeighborsClassifier(n_neighbors=3)

knn_clf.fit(X_train,y_train)

#predicting on test set
y_pred=knn_clf.predict(X_test)


print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1812
           1       1.00      1.00      1.00      2070
           2       1.00      1.00      1.00      1749
           3       1.00      1.00      1.00      1847
           4       1.00      1.00      1.00      1730
           5       1.00      1.00      1.00      1591
           6       1.00      1.00      1.00      1761
           7       1.00      1.00      1.00      1869
           8       1.00      1.00      1.00      1787
           9       1.00      1.00      1.00      1784

   micro avg       1.00      1.00      1.00     18000
   macro avg       1.00      1.00      1.00     18000
weighted avg       1.00      1.00      1.00     18000
'''



#predicting on validation set
y_valid_pred=knn_clf.predict(validate_X)
print(classification_report(validate_y,y_valid_pred))
print(accuracy_score(validate_y,y_valid_pred))


'''
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       980
           1       0.00      0.00      0.00      1135
           2       0.00      0.00      0.00      1032
           3       0.00      0.00      0.00      1010
           4       0.00      0.00      0.00       982
           5       0.00      0.00      0.00       892
           6       0.00      0.00      0.00       958
           7       0.00      0.00      0.00      1028
           8       0.00      0.00      0.00       974
           9       0.00      0.00      0.00      1009

   micro avg       0.00      0.00      0.00     10000
   macro avg       0.00      0.00      0.00     10000
weighted avg       0.00      0.00      0.00     10000

'''
