#APPLICAZIONE DEGLI ALGORITMI CHE FANNO USO DEL SOLO TRAFFICO BENIGNO PER SETTARE LA SOGLIA: ALG1, ALG2, ALG3, N.A.T.D.
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




#ALGORITMI PER IL SETTAGGIO DELLA SOGLIA DI ANOMALIA
def algoritmo1(RMSE):

    global s1
    global s2
    global freq
    count = 0
    
    weights = np.arange(0,1,1/len(RMSE))
    if len(weights)>len(RMSE):
        weights = weights[:len(RMSE)]

    error = RMSE*weights
    s1 = np.mean(error)
    s2 = s1+np.std(error)

    for i in RMSE:
        if i>=s1 and i<=s2:
            count=count+1;

    freq = count/len(RMSE)



def algoritmo2(RMSE):

    RMSE = np.sort(RMSE)
    s = np.percentile(RMSE,75)

    return s



def algoritmo3(RMSE):

    RMSE = np.sort(RMSE)
    p = int((len(RMSE)*70)/100)
    RMSE = RMSE[p:]
    s = mode(RMSE)

    return s



def naive_anomaly_threshold_with_decay(RMSE):

    s= -1
    a=0.9
    
    if np.amax(RMSE)>s:
        s = np.amax(RMSE)
    else:
        s = s*a

    return s




def StampaValori(device, iteration, index, labels, RMSE, algoritmo):
   dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,1], 'TipologiaAttacco': labels[:,2],'RMSE:': RMSE})

   if not os.path.isdir('./Risultati/'+device+'/'+algoritmo):
      os.makedirs('./Risultati/'+device+'/'+algoritmo)

   dataset.to_parquet('./Risultati/'+device+'/'+algoritmo+'/SKF'+str(iteration)+'.parquet',index=False)





os.chdir('/home/francesco/Scrivania/Codice/')
device = Device(0)
dataset_train = dict()
dataset_test = dict()
s1=0
s2=0
freq=0
s=0
count = 0

if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))

tempi = open("Soglie.txt", "a")
tempi.write("\n\n\n%s" %device.name)
tempi.close()


#ALGORITMI CHE UTILIZZANO SOLO DATI BENIGNI PER SETTARE LA SOGLIA
for tss_iteration in range(10):
    path_train = './SKF/'+device.name+'/Train/SKF'+str(tss_iteration)
    path_test = './SKF/'+device.name+'/Test/SKF'+str(tss_iteration)


    dataset_train = pd.read_parquet(path_train+'/SKF4.parquet')
    dataset_train = pd.DataFrame(dataset_train)
    dataset_train = dataset_train.to_numpy().astype('float32')
    dataset_test = pd.read_parquet(path_test+'/SKF4.parquet')
    dataset_test = pd.DataFrame(dataset_test)
    dataset_test = dataset_test.to_numpy().astype('float32')


    #APPLICARE L'ALGORITMO PER IL SETTAGGIO DELLA SOGLIA
    for alg in ['algoritmo1','algoritmo2','algoritmo3','naive_anomaly_threshold_with_decay']:
        if alg == "algoritmo1":
            print("Sto valutando l'algoritmo 1")
            algoritmo1(dataset_train[:,4])

            for j in range(len(dataset_test[:,4])):
                if dataset_test[j,4]>s2:
                    dataset_test[j,1]=1
                elif dataset_test[j,4]>=s1 and dataset_test[j,4]<=s2:
                    count = count+1
                    if count/len(dataset_test[:,4])>freq:
                        dataset_test[j,1]=1
                    else:
                        dataset_test[j,1]=0
                else:
                    dataset_test[j,1]=0

        else:

            if alg == "algoritmo2":
                print("Sto valutando l'algoritmo 2")
                s = algoritmo2(dataset_train[:,4])
            elif alg == "algoritmo3":
                print("Sto valutando l'algoritmo 3")
                s = algoritmo3(dataset_train[:,4])
            elif alg == "naive_anomaly_threshold_with_decay":
                print("Sto valutando l'algoritmo naive anomaly threshold with decay")
                s = naive_anomaly_threshold_with_decay(dataset_train[:,4])


            for j in range(len(dataset_test[:,4])):
                if dataset_test[j,4]>s:
                    dataset_test[j,1] = 1
                else:
                    dataset_test[j,1] = 0


        labels = dataset_test[:,1:4]
        RMSE = dataset_test[:,4]

        StampaValori(device.name, tss_iteration, dataset_test[:,0], labels, RMSE, alg)


