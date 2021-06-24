#APPLICAZIONE DEGLI ALGORITMI STOCHASTIC ANOMALY THRESHOLD NELLE VARIANTI CON 1, 10, 50 E TUTTI I PACCHETTI DI MIRAI
import glob
import pandas as pd
import sys
import math
import datetime
import os
import progressbar
import numpy as np
from statistics import mode
import tensorflow as tf
import time
from pandas import DataFrame
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum
import pickle

import warnings
warnings.filterwarnings("ignore")

class Device(Enum):
   Danmini_Doorbell = 0
   Ecobee_Thermostat = 1
   Ennio_Doorbell = 2
   Philips_B120N10_Baby_Monitor = 3
   Provision_PT_737E_Security_Camera = 4
   Provision_PT_838_Security_Camera = 5
   Samsung_SNH_1011_N_Webcam = 6
   SimpleHome_XCS7_1002_WHT_Security_Camera = 7
   SimpleHome_XCS7_1003_WHT_Security_Camera = 8

class Attack(Enum):
   benign_traffic = 0
   gafgyt_combo = 1
   gafgyt_junk = 2
   gafgyt_scan = 3
   gafgyt_tcp = 4
   gafgyt_udp = 5
   mirai_ack = 6
   mirai_scan = 7
   mirai_syn = 8
   mirai_udp = 9
   mirai_udpplain = 10




def stochastic_anomaly_threshold(RMSE, RMSE_mirai, labels_mirai, training_labels_ben):

    RMSE_all = np.concatenate([RMSE,RMSE_mirai], axis=0)
    training_labels_all = np.concatenate([training_labels_ben, labels_mirai], axis=0)

    s = np.mean(RMSE)+3*np.std(RMSE)
    s_w = s
    acc_w = -1
    v=0.001
    pred = np.zeros(len(RMSE_all))
    tipologia = np.zeros(len(RMSE_all))
    for j in range(len(RMSE_all)):
        if j<len(RMSE): tipologia[j]=0
        else: tipologia[j]=1

    while s>np.mean(RMSE):

        for i in range(0,len(RMSE_all)):
            if RMSE_all[i]>s:
                pred[i] = 1
            else:
                pred[i] = 0

        acc = metrics.accuracy_score(tipologia,pred)

        if acc>acc_w:
            s_w = s
            acc_w = acc

        s = s-v
    s = s_w

    return s





def StampaValori(device, iteration, index, labels, RMSE,algoritmo):
   dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,1], 'TipologiaAttacco': labels[:,2],'RMSE:': RMSE})

   if not os.path.isdir('./Risultati/'+device+'/'+algoritmo):
      os.makedirs('./Risultati/'+device+'/'+algoritmo)

   dataset.to_parquet('./Risultati/'+device+'/'+algoritmo+'/SKF'+str(iteration)+'.parquet',index=False)



os.chdir('/home/francesco/Scrivania/Codice/')
device = Device(0)
dataset_ben = dict()
dataset_mal = dict()
dataset_test = dict()
s=0

if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))


for tss_iteration in range(10):
    path_ben = './SKF/'+device.name+'/Train/SKF'+str(tss_iteration)
    path_mal = './SKF/'+device.name+'/Train_mal/SKF'+str(tss_iteration)
    path_test = './SKF/'+device.name+'/Test/SKF'+str(tss_iteration)


    dataset_ben = pd.read_parquet(path_ben+'/SKF4.parquet')
    dataset_ben = pd.DataFrame(dataset_ben)
    dataset_ben = dataset_ben.to_numpy().astype('float32')

    dataset_mal = pd.read_parquet(path_mal+'/SKF4.parquet')
    dataset_mal = pd.DataFrame(dataset_mal)
    dataset_mal = dataset_mal.to_numpy().astype('float32')

    dataset_test = pd.read_parquet(path_test+'/SKF4.parquet')
    dataset_test = pd.DataFrame(dataset_test)
    dataset_test = dataset_test.to_numpy().astype('float32')


    for alg in ["stochastic_anomaly_threshold_1","stochastic_anomaly_threshold_10","stochastic_anomaly_threshold_50","stochastic_anomaly_threshold_all"]:

        if alg == "stochastic_anomaly_threshold_1":
            print("Utilizzo 1 campione di Mirai")
            RMSE = dataset_ben[:,4]  
            labels = dataset_ben[:,1]

            RMSE_mirai_scan = np.zeros(dataset_mal.shape[0])
            labels_mirai_scan = np.zeros(dataset_mal.shape[0])
            pos = 0
           #seleziono solo l'attacco mirai scan
            for j in range(0,len(dataset_mal[:,1])):
                if dataset_mal[j,3]==7:
                    RMSE_mirai_scan[pos]=dataset_mal[j,4]
                    labels_mirai_scan[pos]=1
                    pos = pos+1

            RMSE_mirai_scan = RMSE_mirai_scan[:1]
            labels_mirai_scan = labels_mirai_scan[:1]
            #print(RMSE_mirai_scan)
            #print(labels_mirai_scan)
            input()
    
            s = stochastic_anomaly_threshold(RMSE, RMSE_mirai_scan, labels_mirai_scan, labels)
             

        elif alg == "stochastic_anomaly_threshold_10":
            print("Utilizzo 10 campioni di Mirai")
            RMSE = dataset_ben[:,4]  
            labels = dataset_ben[:,1]

            RMSE_mirai_scan = np.zeros(dataset_mal.shape[0])
            labels_mirai_scan = np.zeros(dataset_mal.shape[0])
            pos = 0
           #seleziono solo l'attacco mirai scan
            for j in range(0,len(dataset_mal[:,1])):
                if dataset_mal[j,3]==7:
                    RMSE_mirai_scan[pos]=dataset_mal[j,4]
                    labels_mirai_scan[pos]=1
                    pos = pos+1

            RMSE_mirai_scan = RMSE_mirai_scan[:10]
            labels_mirai_scan = labels_mirai_scan[:10]
            #print(RMSE_mirai_scan)
            #print(labels_mirai_scan)
            input()
    
            s = stochastic_anomaly_threshold(RMSE, RMSE_mirai_scan, labels_mirai_scan, labels)

    
        elif alg == "stochastic_anomaly_threshold_50":
            print("Utilizzo 50 campioni di Mirai")
            RMSE = dataset_ben[:,4]  
            labels = dataset_ben[:,1]

            RMSE_mirai_scan = np.zeros(dataset_mal.shape[0])
            labels_mirai_scan = np.zeros(dataset_mal.shape[0])
            pos = 0
           #seleziono solo l'attacco mirai scan
            for j in range(0,len(dataset_mal[:,1])):
                if dataset_mal[j,3]==7:
                    RMSE_mirai_scan[pos]=dataset_mal[j,4]
                    labels_mirai_scan[pos]=1
                    pos = pos+1

            RMSE_mirai_scan = RMSE_mirai_scan[:50]
            labels_mirai_scan = labels_mirai_scan[:50]
            #print(RMSE_mirai_scan)
            #print(labels_mirai_scan)
            input()
    
            s = stochastic_anomaly_threshold(RMSE, RMSE_mirai_scan, labels_mirai_scan, labels)



        elif alg== "stochastic_anomaly_threshold_all":
            print("Utilizzo tutto il traffico Mirai")
            RMSE = dataset_ben[:,4]  
            labels = dataset_ben[:,1]

            RMSE_mirai = np.zeros(dataset_mal.shape[0])
            labels_mirai = np.zeros(dataset_mal.shape[0])
            pos = 0
            for j in range(0,len(dataset_mal[:,4])):
                if dataset_mal[j,3]>=6 and dataset_mal[j,3]<=10:
                    RMSE_mirai[pos]=dataset_mal[j,4]
                    labels_mirai[pos] = 1
                    pos =pos+1

            RMSE_mirai=RMSE_mirai[:pos]
            labels_mirai=labels_mirai[:pos]
            #print(RMSE_mirai)
            #print(labels_mirai)
            input()

            s = stochastic_anomaly_threshold(RMSE, RMSE_mirai, labels_mirai, labels)


        for j in range(len(dataset_test[:,4])):
            if dataset_test[j,4]>s:
                dataset_test[j,1] = 1
            else:
                dataset_test[j,1] = 0


        labels_final = dataset_test[:,1:4]
        RMSE_final = dataset_test[:,4]

        StampaValori(device.name, tss_iteration, dataset_test[:,0], labels_final, RMSE_final, alg)


