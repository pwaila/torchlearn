'''
@author: Pranav
This is simple neural neural network, which works on well known IRIS data.
'''

import torch as T
import torch 
import torchvision as TV 
import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split



#Loading iris dataset from sklearn
data=load_iris()

#spliting data in train and test set
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.33, random_state=42)

# creating a simple neural network model with 3 linear layers
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

h1= 20
h2=10

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=nn.Linear(4,h1)
        self.l2=nn.Linear(h1,h2)
        self.l3=nn.Linear(h2,3)

    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        return x

#loading model
net=Net()

#defining loss function and optimization algorithm
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer= optim.SGD(net.parameters(),lr=.001,momentum=.9)

#number of epoch for training the nework
num_epoch = 5000

for epoch in range(num_epoch):
    #creating training variables in placeholders
    X = Variable(T.Tensor(X_train).float())
    Y = Variable(T.Tensor(y_train).long())
    
    #initializing optimizer
    optimizer.zero_grad()
    out=net(X)
    loss=criterion(out,Y)
    loss.backward()
    optimizer.step()

    # print statistics
    if (epoch) % 50 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' %(epoch+1, 2, loss.data))

print('finish')

#############
#prediction

#get prediction
X = Variable(T.Tensor(X_test).float())
Y = T.Tensor(y_test).long()

out = net(X)
predict = T.argmax(out.data,dim=1)


#get accuration
print('Accuracy of the network %d %%' % (100 * T.sum(Y==predict) / 50)) 