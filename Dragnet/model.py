# Official InceptionTime implementation: https://github.com/hfawaz/InceptionTime 
# Unofficial pytorch implementation of InceptionTime: https://github.com/TheMrGhostman/InceptionTime-Pytorch
# Fuzzy cmeans: https://www.kaggle.com/prateekk94/fuzzy-c-means-clustering-on-iris-dataset

#imports
import torch
import pandas as pd
import copy

import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import numpy as np 
import time
import torch.nn.functional as F 
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import operator
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

History_len=25
Future_len=300-History_len
k=15

def getClusterHeadDict(Centers,l,k):
  Dict={}
  for i in range(k):
    Dict[i]=[Centers[i,:l],Centers[i,l:]]
  return Dict
   
class Flatten(nn.Module):
  def __init__(self, out_features):
    super(Flatten, self).__init__()
    self.output_dim = out_features

  def forward(self, x):
    return x.view(-1, self.output_dim)  
    
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

class ReshapeDecode(nn.Module):
  def __init__(self, out_shape1,out_shape2):
    super(ReshapeDecode, self).__init__()
    self.out_shape1 = out_shape1
    self.out_shape2 = out_shape2

  def forward(self, x):
    return x.view(-1, self.out_shape1, self.out_shape2)

def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def pass_through(X):
	return X

class InceptionPast(nn.Module):
  def __init__(self,Sequence_length,in_channels, n_filters, kernel_sizes, bottleneck_channels=32, activation=nn.ReLU(), return_indices=True):
    super(InceptionPast, self).__init__()
    self.return_indices=return_indices
    self.bottleneck = pass_through
    bottleneck_channels = 1
    self.seriesHistory_len=Sequence_length

    self.conv_from_bottleneck_1 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[0], 
                    stride=1, 
                    padding=kernel_sizes[0]//2, 
                    bias=False
                    )
    self.conv_from_bottleneck_2 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[1], 
                    stride=1, 
                    padding=kernel_sizes[1]//2, 
                    bias=False
                    )
    self.conv_from_bottleneck_3 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[2], 
                    stride=1, 
                    padding=kernel_sizes[2]//2, 
                    bias=False
                    )
    self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
    self.conv_from_maxpool = nn.Conv1d(
                  in_channels=in_channels, 
                  out_channels=n_filters, 
                  kernel_size=1, 
                  stride=1,
                  padding=0, 
                  bias=False
                  )
    self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
    self.activation = activation
    self.AvgPool1d=nn.AdaptiveAvgPool1d(output_size=1)
    self.FlatLayer=nn.Flatten()
    self.FC1=nn.Linear(in_features=4*32*self.seriesHistory_len, out_features=4*32*10)
    self.FC2=nn.Linear(in_features=4*32*10, out_features=4*32)
    self.LatentFeatures=nn.Linear(in_features=4*32, out_features=4*8)

  def forward(self, X):
      #print('In inception')
      # step 1
      Z_bottleneck = self.bottleneck(X)
      #print(Z_bottleneck.shape)
      if self.return_indices:
        Z_maxpool, indices = self.max_pool(X)
      else:
        Z_maxpool = self.max_pool(X)
      Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
      Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
      Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
      Z4 = self.conv_from_maxpool(Z_maxpool)
      Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
      Z = self.activation(self.batch_norm(Z))
      Z = self.FlatLayer(Z) 
      Z = self.FC1(Z) 
      Z = self.FC2(Z) 
      Z = self.LatentFeatures(Z)
      if self.return_indices:
        return Z, indices
      else:
        return Z

class InceptionFuture(nn.Module):
  def __init__(self,Sequence_length,in_channels, n_filters, kernel_sizes, bottleneck_channels=32, activation=nn.ReLU(), return_indices=True):
    super(InceptionFuture, self).__init__()
    self.return_indices=return_indices
    self.bottleneck = pass_through
    bottleneck_channels = 1
    self.seriesFuture_len=Sequence_length

    self.conv_from_bottleneck_1 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[0], 
                    stride=1, 
                    padding=kernel_sizes[0]//2, 
                    bias=False
                    )
    self.conv_from_bottleneck_2 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[1], 
                    stride=1, 
                    padding=kernel_sizes[1]//2, 
                    bias=False
                    )
    self.conv_from_bottleneck_3 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[2], 
                    stride=1, 
                    padding=kernel_sizes[2]//2, 
                    bias=False
                    )
    self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
    self.conv_from_maxpool = nn.Conv1d(
                  in_channels=in_channels, 
                  out_channels=n_filters, 
                  kernel_size=1, 
                  stride=1,
                  padding=0, 
                  bias=False
                  )
    self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
    self.activation = activation
    self.AvgPool1d=nn.AdaptiveAvgPool1d(output_size=1)
    self.FlatLayer=nn.Flatten()
    self.FC1=nn.Linear(in_features=4*32*self.seriesFuture_len, out_features=4*32*50)
    self.FC2=nn.Linear(in_features=4*32*50, out_features=4*32*5)
    self.LatentFeatures=nn.Linear(in_features=4*32*5, out_features=4*32)

  def forward(self, X):
      Z_bottleneck = self.bottleneck(X)
      if self.return_indices:
        Z_maxpool, indices = self.max_pool(X)
      else:
        Z_maxpool = self.max_pool(X)
      Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
      Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
      Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
      Z4 = self.conv_from_maxpool(Z_maxpool)
      Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
      Z = self.activation(self.batch_norm(Z))
      Z = self.FlatLayer(Z) 
      Z = self.FC1(Z) 
      Z = self.FC2(Z) 
      Z = self.LatentFeatures(Z)
      if self.return_indices:
        return Z, indices
      else:
        return Z

class InceptionTranspose(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
    super(InceptionTranspose, self).__init__()
    self.activation = activation
    self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
                    in_channels=in_channels, 
                    out_channels=bottleneck_channels, 
                    kernel_size=kernel_sizes[0], 
                    stride=1, 
                    padding=kernel_sizes[0]//2, 
                    bias=False
                    )
    self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
                    in_channels=in_channels, 
                    out_channels=bottleneck_channels, 
                    kernel_size=kernel_sizes[1], 
                    stride=1, 
                    padding=kernel_sizes[1]//2, 
                    bias=False
                    )
    self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
                    in_channels=in_channels, 
                    out_channels=bottleneck_channels, 
                    kernel_size=kernel_sizes[2], 
                    stride=1, 
                    padding=kernel_sizes[2]//2, 
                    bias=False
                    )
    self.conv_to_maxpool = nn.Conv1d(
                  in_channels=in_channels, 
                  out_channels=out_channels, 
                  kernel_size=1, 
                  stride=1,
                  padding=0, 
                  bias=False
                  )
    self.max_unpool1 = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
    self.max_unpool2 = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
    self.bottleneck = nn.Conv1d(
                in_channels=3*bottleneck_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1, 
                bias=False
                )
    self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
    self.FC1_dec=nn.Linear(in_features=(4*8+4*32), out_features=4*32*10)
    self.FC2_dec=nn.Linear(in_features=4*32*10, out_features=4*32*50)
    self.FC3_dec=nn.Linear(in_features=4*32*50, out_features=4*32*300)
    self.reshape =ReshapeDecode(128,300)
    self.Flat =nn.Flatten()
  
  def forward(self, X):
    Ind1=X[1]
    Ind2=X[2]
    X = self.FC1_dec(X[0])
    X = self.FC2_dec(X)
    X = self.FC3_dec(X)
    X = self.reshape(X)
    Z1 = self.conv_to_bottleneck_1(X)
    Z2 = self.conv_to_bottleneck_2(X)
    Z3 = self.conv_to_bottleneck_3(X)
    Z4 = self.conv_to_maxpool(X)

    Z = torch.cat([Z1, Z2, Z3], axis=1)
    Z4_list=torch.split(Z4,[History_len,Future_len], dim=-1)
    MUP1 = self.max_unpool1(Z4_list[0],Ind1 )
    MUP2 = self.max_unpool1(Z4_list[1],Ind2 )
    MUP=torch.cat((MUP1,MUP2), dim=-1)
    BN = self.bottleneck(Z)
    return self.Flat(self.activation(self.batch_norm(BN + MUP)))

def calc_distance(X1, X2,past_len):
    return (((sum((X1[:past_len] - X2[:past_len])**2))**0.5)*0.7 + ((sum((X1[past_len:] - X2[past_len:])**2))**0.5)*0.3)

def assign_clusters(centroids, cluster_array,past_len):
    clusters = []
    for i in range(cluster_array.shape[0]):
        distances = []
        for centroid in centroids:
            distances.append(calc_distance(centroid, 
                                           cluster_array[i],past_len))
        cluster = [z for z, val in enumerate(distances) if val==min(distances)]
        clusters.append(cluster[0])
    return clusters

def calc_centroids(clusters, cluster_array):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, 
                                         columns=['cluster'])], 
                           axis=1)
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']\
                                     ==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

def KMeanSpecial(k,cluster_array,pl):
  cluster_vars = []
  centroids = [cluster_array[i+2] for i in range(k)]
  clusters = assign_clusters(centroids, cluster_array,pl)
  initial_clusters = clusters
  for i in tqdm(range(100)):
    prev_centroids= centroids
    centroids = calc_centroids(clusters, cluster_array)
    clusters = assign_clusters(centroids, cluster_array,pl)
    if checkConvergence(prev_centroids,centroids):
      print(i)
      break
  return clusters,centroids

def checkConvergence(prev_centroids,centroids):
  if (np.asarray(prev_centroids) == np.asarray(centroids)).all() :
    return 1
  else:
    return 0

class DeepFeedforward_MultiLabels_senti(torch.nn.Module):
        def __init__(self, input_size,input_size_senti, L1,L2,L3,sL1,sL2,num_classes):
            super(DeepFeedforward_MultiLabels_senti, self).__init__()
            self.input_size = input_size
            self.L1  = L1
            self.L2  = L2
            self.L3  = L3
            self.input_size_senti = input_size_senti
            self.sL1  = sL1
            self.sL2  = sL2
            self.classes_num = num_classes
            self.fc1 = torch.nn.Linear(self.input_size, self.L1)
            self.relu1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.L1, self.L2)
            self.relu2 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(self.L2, self.L3)
            self.relu3 = torch.nn.ReLU()

            self.Sfc1 = torch.nn.Linear(self. input_size_senti, self.sL1)
            self.Srelu1 = torch.nn.ReLU()
            self.Sfc2 = torch.nn.Linear(self.sL1, self.sL2)
            self.Srelu2 = torch.nn.ReLU()
           

            self.fc4 = torch.nn.Linear((self.L3+self.sL2),self.classes_num)
            self.sig = torch.nn.Sigmoid()
        def forward(self, X):
            x_past = X[0]
            X_sent = X[1]
            hidden1 = self.fc1(x_past)
            Relu1 = self.relu1(hidden1)
            hidden2 = self.fc2(Relu1)
            Relu2 = self.relu2(hidden2)
            hidden3 = self.fc3(Relu2)
            Relu3 = self.relu3(hidden3)

            shidden1 = self.Sfc1(X_sent)
            sRelu1 = self.Srelu1(shidden1)
            shidden2 = self.Sfc2(sRelu1)
            sRelu2 = self.Srelu2(shidden2)

            X_final = torch.cat((Relu3,sRelu2), dim=-1)

            output = self.fc4(X_final)
            output = self.sig(output)
            return output

class Prediction_FC(torch.nn.Module):
        def __init__(self, input_size_Past,input_size_Future, L1,L2,L3,L4):
            super(Prediction_FC, self).__init__()
            self.input_size_Past = input_size_Past
            self.input_size_Future = input_size_Future
            self.L1  = L1
            self.L2  = L2
            self.L3  = L3
            self.L4  = L4
            self.fc1 = torch.nn.Linear(self.input_size_Past, self.L1)
            self.sig0 = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear((self.L1+self.input_size_Past+self.input_size_Future), self.L2)
            self.relu1 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(self.L2,self.L3)
            self.relu2 = torch.nn.ReLU()
            self.fc4 = torch.nn.Linear(self.L3,self.L4)
            self.relu3 = torch.nn.ReLU()
        def forward(self, X):
            x_past=X[0]
            xc_past=X[1]
            xc_future=X[2]
            past_prior = x_past - xc_past
            Xp = self.fc1(past_prior)
            Xp = self.sig0(Xp)
            Xp = torch.cat((x_past,Xp), dim=-1)
            Xf=torch.cat((Xp,xc_future), dim=-1)
            Xf = self.fc2(Xf)
            Xf = self.fc3(Xf)
            Xf = self.fc4(Xf)
            return Xf

Encoder_past=nn.Sequential(
                    Reshape(out_shape=(1,History_len)),
                    InceptionPast(
                        Sequence_length=History_len,
                        in_channels=1, 
                        n_filters=32, 
                        kernel_sizes=[5,7,9],
                        bottleneck_channels=32,
                        activation=nn.ReLU()
                    )
        ).to(device)

Encoder_future=nn.Sequential(
                    Reshape(out_shape=(1,Future_len)),
                    InceptionFuture(
                        Sequence_length=Future_len,
                        in_channels=1, 
                        n_filters=32, 
                        kernel_sizes=[5,7,9],
                        bottleneck_channels=32,
                        activation=nn.ReLU()
                    )
        ).to(device)

decoder=InceptionTranspose(
                        in_channels=128, 
                        out_channels=1, 
                        kernel_sizes=[5,7,9],
                        bottleneck_channels=32,
                        activation=nn.ReLU()
                        ).to(device)

Classifier = DeepFeedforward_MultiLabels_senti(32,History_len,50,40,30,15,5,k).to(device)

Predictor=Prediction_FC(32,128,12,200,160,128).to(device)

senti_time_series=pickle.load(open('sentiment_series.pkl', 'rb'))
final_time_series=pickle.load(open('time_series.pkl', 'rb'))


senti_time_series_t = []
for i in final_time_series:
    for j in senti_time_series:
        if i[-1] == j[-1]:
            senti_time_series_t.append(j)

senti_time_series = senti_time_series_t

train_data=[]
max_length=0
for data in final_time_series:
  if len(data[0]) > max_length:
    max_length=len(data[0])
  train_data.append(data[0])

#normalised by length
train_data=[]
max_length=0
for data in final_time_series:
  if len(data[0]) > max_length:
    max_length=len(data[0])
  train_data.append(data[0]/np.max(data[0]))

max_length=300

TimeSeries=train_data
train_data=[]
pad=0
cut=0
for x in TimeSeries:
    if len(x)<max_length:
      pad=pad+1
      train_data.append(np.pad(x, (0,(max_length-len(x))), 'constant', constant_values=(0)))
    else:
      cut=cut+1
      train_data.append(x[0:max_length])

TimeSeries=train_data

TimeSeries=np.asarray(TimeSeries)
training, test = TimeSeries[:2760,:], TimeSeries[2760:,:]

senti_train_data=[]
for data in senti_time_series:
  senti_train_data.append(data[0])

Senti_train_data=[]
for x in senti_train_data:
      Senti_train_data.append(x[0:History_len])
        
Senti_train_data = np.asarray(Senti_train_data)
        
training_senti, test_senti = Senti_train_data[:2760,:], Senti_train_data[2760:,:]

# Encoder-Decoder train
# Store Loss
Trloss=[]

# set optimisers and criterion
EncoderPast_optimizer = torch.optim.SGD(Encoder_past.parameters(), lr=0.001)
EncoderFuture_optimizer = torch.optim.SGD(Encoder_future.parameters(), lr=0.001)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.001)

AutoEncoder_criterion = torch.nn.MSELoss() 

#Set model for train
Encoder_past.train()
Encoder_future.train()
decoder.train()
epochs = 500
n_batches=40
for epoch in tqdm(range(epochs)):
  x= shuffle(training)
  Loss=[]
  for i in range(69):
        local_X= x[i*n_batches:(i+1)*n_batches]
        Past_x =local_X[:,:History_len]
        Future_x =local_X[:,History_len:]

        #set optimizers so that prev gradient is cleared
        EncoderPast_optimizer.zero_grad()
        EncoderFuture_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        #latent representation
        XP , ind1 =Encoder_past(torch.tensor(Past_x).float().to(device))
        XF , ind2 =Encoder_future(torch.tensor(Future_x).float().to(device))
        Latent_space=torch.cat((XP,XF), dim=-1)
        #reconstruction
        original_created= decoder([Latent_space,ind1,ind2])
        
        # Loss calcultion
        loss = AutoEncoder_criterion(original_created, torch.tensor(local_X).float().to(device))   

        Loss.append(loss.item())

        # Backward pass
        loss.backward()
        EncoderPast_optimizer.step()
        EncoderFuture_optimizer.step()
        decoder_optimizer.step()
  loss_train=np.mean(np.asarray(Loss))      
  Trloss.append(loss_train)

#model optimizers and criterion
Classifier_optimizer = torch.optim.Adam(Classifier.parameters(), lr=0.001)
Classifier_criterion = torch.nn.BCELoss()

#set to train model
Classifier.train()
Encoder_past.eval()
Encoder_future.eval()

epochs = 2000
n_batches=60

#Store loss
Trloss=[]

#Store accuracy
TrAcc=[]
TeAcc=[]


x= shuffle(training)

#split train data in past and future
Past_x =x[:,:History_len]
Future_x =x[:,History_len:]

#latent representation
XP , ind1 =Encoder_past(torch.tensor(Past_x).float().to(device))
XF , ind2 =Encoder_future(torch.tensor(Future_x).float().to(device))

#concatenate for clustering
X_train=torch.cat((XP,XF), dim=-1)

# apply clustering to get centroids and train_labels
gm = GaussianMixture(n_components=k, random_state=0).fit(X_train.cpu().detach().numpy())
centres =gm.means_
membership=gm.predict_proba(X_train.cpu().detach().numpy())
Dict_of_cluster_heads = getClusterHeadDict(centres,32,k)

for epoch in tqdm(range(epochs)):
      Loss=[]
     
      # shuffle for classifier
      x, y = shuffle(XP,membership)
      Loss=[]
      for i in range(46):
        
        local_X, local_y = x[i*n_batches:(i+1)*n_batches], y[i*n_batches:(i+1)*n_batches]
        Classifier_optimizer.zero_grad()
        

        y_pred = Classifier(torch.tensor(local_X).float().to(device))

        # Loss calcultion
        loss = Classifier_criterion(y_pred, torch.tensor(local_y).float().to(device))
        Loss.append(loss.item())   
        
        # Backward pass
        loss.backward()
        Classifier_optimizer.step() 
        
      loss_train=np.mean(np.asarray(Loss))     
      Trloss.append(loss_train)

def getXcp(Dict_of_clusters,memberMatrix):
  XcList=[]
  for id in memberMatrix:
    head = np.zeros((k,32))
    for i in range(k):
      #print("id:",id[i])
      #print('head:',Dict_of_clusters[i][0])
      head[i]=id[i]*Dict_of_clusters[i][0]
    #print('final:',np.mean(head,axis=0))
    XcList.append(np.mean(head,axis=0))
  return np.asarray(XcList)
def getXcf(Dict_of_clusters,memberMatrix):
  XcList=[]
  for id in memberMatrix:
    head = np.zeros((k,128))
    for i in range(k):
      #print("id:",id[i])
      #print('head:',Dict_of_clusters[i][1])
      head[i]=id[i]*Dict_of_clusters[i][1]
    #print('final:',np.mean(head,axis=0))
    XcList.append(np.mean(head,axis=0))
  return np.asarray(XcList)

#set to train mode
Classifier.eval()
Encoder_past.eval()
Encoder_future.eval()
Predictor.train()

#Store loss
Trloss=[]

#model optimizers and criterion
Prediction_FC_optimizer = torch.optim.SGD(Predictor.parameters(), lr=0.01)
predictor_criterion = torch.nn.MSELoss()

epochs = 1000
n_batches=60
for epoch in tqdm(range(epochs)):
      Loss=[]    
      # shuffle for classifier
      x= shuffle(training)
      for i in range(46):
        local_X= x[i*n_batches:(i+1)*n_batches]
        Past_x =local_X[:,:History_len]
        Future_x =local_X[:,History_len:]

        #set optimizers so that prev gradient is cleared
        Prediction_FC_optimizer.zero_grad()

        #latent representation
        XP , ind1 =Encoder_past(torch.tensor(Past_x).float().to(device))
        XF , ind2 =Encoder_future(torch.tensor(Future_x).float().to(device))
        Latent_space=torch.cat((XP,XF), dim=-1)
        membership_mat =gm.predict_proba(Latent_space.cpu().detach().numpy())

        
        #print('Local_X',local_X.shape)
        yf_latent =  Predictor([XP,torch.tensor(getXcp(Dict_of_cluster_heads,membership_mat)).float().to(device),torch.tensor(getXcf(Dict_of_cluster_heads,membership_mat)).float().to(device)])
     
       
        # Loss calcultion
        loss = predictor_criterion(yf_latent,XF)   
        Loss.append(loss.item())   
        
        # Backward pass
        loss.backward()
        Prediction_FC_optimizer.step() 
      loss_train=np.mean(np.asarray(Loss))     
      Trloss.append(loss_train)

with torch.no_grad():
        Past_x =test[:,:History_len]
        Future_x =test[:,History_len:]

        #latent representation
        XP , ind1 =Encoder_past(torch.tensor(Past_x).float().to(device))
        XF , ind2 =Encoder_future(torch.tensor(Future_x).float().to(device))
        Latent_space=torch.cat((XP,XF), dim=-1)
        
membership_mat=gm.predict_proba(Latent_space.cpu().detach().numpy())

yf_latent =  Predictor([XP,torch.tensor(getXcp(Dict_of_cluster_heads,membership_mat)).float().to(device),torch.tensor(getXcf(Dict_of_cluster_heads,membership_mat)).float().to(device)])
Latent_space = torch.cat((torch.tensor(XP).float().to(device),yf_latent), dim=-1)

with torch.no_grad():
   Not_original=decoder([Latent_space,ind1,ind2]).cpu().detach().numpy()

print(Not_original.shape, Future_x.shape)

pickle.dump(Not_original, open('./Not_original.pkl', 'wb'))
pickle.dump(Future_x, open('./Future_x.pkl', 'wb'))
