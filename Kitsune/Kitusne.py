from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)

import glob
import pandas as pd
import sys
import math
import datetime
import os
import progressbar
import numpy as np
from statistics import mode
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




def StampaValoriTrain(device, iteration, index, labels, RMSE, mal,t):
    if mal ==0:
        dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,2], 'TipologiaAttacco': labels[:,1],'RMSE:': RMSE, 'Used': labels[:,3]})

        if not os.path.isdir('./SKF/'+device+'/Train/SKF'+str(iteration)):
            os.makedirs('./SKF/'+device+'/Train/SKF'+str(iteration))
        #dataset.to_csv('./SKF/'+device+'/Train/SKF'+str(iteration)+'/SKF'+str(t)+'.csv',index=False, sep=',')
        dataset.to_parquet('./SKF/'+device+'/Train/SKF'+str(iteration)+'/SKF'+str(t)+'.parquet', index=False)
    else:
        dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,2], 'TipologiaAttacco': labels[:,1],'RMSE:': RMSE})

        if not os.path.isdir('./SKF/'+device+'/Train_mal/SKF'+str(iteration)):
            os.makedirs('./SKF/'+device+'/Train_mal/SKF'+str(iteration))

        dataset.to_parquet('./SKF/'+device+'/Train_mal/SKF'+str(iteration)+'/SKF'+str(t)+'.parquet',index=False)



def StampaValoriTest(device, iteration, index, labels, RMSE,t):
   dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,2], 'TipologiaAttacco': labels[:,1],'RMSE:': RMSE})

   if not os.path.isdir('./SKF/'+device+'/Test/SKF'+str(iteration)):
      os.makedirs('./SKF/'+device+'/Test/SKF'+str(iteration))

   dataset.to_parquet('./SKF/'+device+'/Test/SKF'+str(iteration)+'/SKF'+str(t)+'.parquet',index=False)


def load_clusters():
   clusters = dict()
   clusters_path = './FPMaXX/'

   clusters['Base'] = dict()


   clusters['Base'][0] = range(0,23)
   clusters['Base'][1] = range(23,46)
   clusters['Base'][2] = range(46,69)
   clusters['Base'][3] = range(69,92)
   clusters['Base'][4] = range(92,115)

   return clusters



def load_dataset(dev, cat):
    dataset = dict()
    progress = 0

    if cat==0:
        dataset_path = './Dataset/'
        csv_paths = glob.glob(dataset_path+dev.name+'/*.csv', recursive = True)

        print("Loading benign dataset")
        bar = progressbar.ProgressBar(maxval=len(csv_paths), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for csv_path in csv_paths:
            attack = ''
            device = csv_path.split('/')[2]

            if device not in dataset:
                dataset[device] = {}
            if( len(csv_path.split('/')) == 4):
                attack = csv_path.split('/')[3]
            else:
                attack = csv_path.split('/')[4]

            attack = attack.replace(".csv","")
            dataset[attack] = pd.read_csv(csv_path, delimiter = ',')
            dataset[attack]['Malign'] = 0
            dataset[attack]['Attack'] = 0
            dataset[attack]['Device'] = Device[device].value
            dataset[attack]['Use'] = 0
            progress += 1
            bar.update(progress)
        bar.finish()
    else:
        dataset_path = './Dataset/'
        csv_paths_mirai = glob.glob(dataset_path+dev.name+'/mirai/*.csv')
        csv_paths_gafgyt = glob.glob(dataset_path+dev.name+'/gafgyt/*.csv')
        csv_paths = csv_paths_mirai+csv_paths_gafgyt

        print("Loading malign dataset")
        bar = progressbar.ProgressBar(maxval=len(csv_paths), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for csv_path in csv_paths:
            attack = ''
            device = csv_path.split('/')[2]
            
            if( len(csv_path.split('/')) == 5):
                attack = csv_path.split('/')[3]+'_'+csv_path.split('/')[4]
            else:
                attack = csv_path.split('/')[2]

            
            attack = attack.replace(".csv","")
            dataset[attack] = pd.read_csv(csv_path, delimiter = ',')
            dataset[attack]['Malign'] = 1
            dataset[attack]['Attack'] = Attack[attack].value
            dataset[attack]['Device'] = Device[device].value
            dataset[attack]['Use'] = 0
            progress += 1
            bar.update(progress)
        bar.finish()
    return dataset





os.chdir('/home/francesco/Scrivania/Codice/')
device = Device(0)
device_dataset = dict()


if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))


benign_dataset = load_dataset(device, 0)
benign_dataset=pd.DataFrame(benign_dataset['benign_traffic'])
benign_dataset = pd.concat([benign_dataset], ignore_index=True)
benign_dataset = benign_dataset.iloc[2048:]
malign_dataset = load_dataset(device, 1)


if int(sys.argv[1])==2 or int(sys.argv[1])==6:
    malign_dataset5 = pd.DataFrame(malign_dataset['gafgyt_udp'])
    malign_dataset6 = pd.DataFrame(malign_dataset['gafgyt_combo'])
    malign_dataset7 = pd.DataFrame(malign_dataset['gafgyt_junk'])
    malign_dataset8 = pd.DataFrame(malign_dataset['gafgyt_scan'])
    malign_dataset9 = pd.DataFrame(malign_dataset['gafgyt_tcp'])
    malign_dataset = pd.concat([malign_dataset5,malign_dataset6,malign_dataset7,malign_dataset8,malign_dataset9], ignore_index=True)
else:
    malign_dataset0 = pd.DataFrame(malign_dataset['mirai_udpplain'])
    malign_dataset1 = pd.DataFrame(malign_dataset['mirai_udp'])
    malign_dataset2 = pd.DataFrame(malign_dataset['mirai_ack'])
    malign_dataset3 = pd.DataFrame(malign_dataset['mirai_scan'])
    malign_dataset4 = pd.DataFrame(malign_dataset['mirai_syn'])
    malign_dataset5 = pd.DataFrame(malign_dataset['gafgyt_udp'])
    malign_dataset6 = pd.DataFrame(malign_dataset['gafgyt_combo'])
    malign_dataset7 = pd.DataFrame(malign_dataset['gafgyt_junk'])
    malign_dataset8 = pd.DataFrame(malign_dataset['gafgyt_scan'])
    malign_dataset9 = pd.DataFrame(malign_dataset['gafgyt_tcp'])
    malign_dataset = pd.concat([malign_dataset0,malign_dataset1,malign_dataset2,malign_dataset3,malign_dataset4,malign_dataset5,malign_dataset6,malign_dataset7,malign_dataset8,malign_dataset9], ignore_index=True)


dataset_all = pd.concat([benign_dataset,malign_dataset], ignore_index=True)

benign_dataset = benign_dataset.to_numpy()
benign_dataset = benign_dataset.astype('float32')
malign_dataset = malign_dataset.to_numpy()
malign_dataset = np.array(malign_dataset, dtype='float32')
benign_dataset = np.concatenate([benign_dataset], axis = 0)


dataset_all = dataset_all.to_numpy()
np.random.shuffle(dataset_all)
clusters = load_clusters()



skf = StratifiedKFold(n_splits = 10, shuffle = False)

device = device.name


print("\nTraining Base")
n_autoencoder = 5
tss_iteration = 0

for train_index, test_index in skf.split(dataset_all, dataset_all[:,116]):
    with tf.device('/cpu:0'):
        print("Train:", train_index, "Test:", test_index)
        train_index = train_index.astype('int32')
        test_index = test_index.astype('int32')
        training = dataset_all[train_index, :119]

        #dataset di addestramento benigno
        training_ben = training[(training[:,115] == 0)]
        np.random.shuffle(training_ben)
        training_features_ben = training_ben[:,:115]
        training_labels_ben = training_ben[:, 115:119]
        train_index_ben = [i for i in train_index if dataset_all[i,115]==0]
        training_labels_ben = training_labels_ben.astype('int')

        #dataset di addestramento malevolo
        training_mal = training[(training[:,115] != 0)]
        training_features_mal = training_mal[:,:115]
        training_labels_mal = training_mal[:, 115:119]
        train_index_mal = [i for i in train_index if dataset_all[i,115]!=0]
        training_labels_mal = training_labels_mal.astype('int')

        #dataset di test
        testing = dataset_all[test_index, : 118]
        test_features = dataset_all[test_index, : 115]
        test_labels = dataset_all[test_index, 115:119]
        test_labels = test_labels.astype('int')




        #COSTRUZIONE MODELLO KITSUNE
        Ensemble = np.empty(n_autoencoder, dtype = object)

        #Building autoencoders & output
        for i, (cluster_number, cluster_elements) in enumerate(clusters['Base'].items()): 
            #index, (key, value). i = numero ordinato del cluster, cluster_number = numero assegnato dall'algoritmo (inutile), cluster_elements = lista delle features nel cluster
            n_cluster_elements = len(cluster_elements)
            Ensemble[i]= Sequential()
            Ensemble[i].add(Dense(units=n_cluster_elements,activation='relu',input_shape=(n_cluster_elements,)))
            Ensemble[i].add(Dense(units=math.ceil(0.75*n_cluster_elements),activation='relu'))
            Ensemble[i].add(Dense(units=n_cluster_elements,activation='sigmoid'))
            Ensemble[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

        Output= Sequential()
        Output.add(Dense(units=n_autoencoder,activation='relu',input_shape=(n_autoencoder,)))
        Output.add(Dense(units=math.ceil(0.75*n_autoencoder),activation='relu'))
        Output.add(Dense(units=n_autoencoder,activation='sigmoid'))
        Output.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
        scaler1=MinMaxScaler(feature_range=(0,1))




        #ADDESTRAMENTO: PARTE BENIGNA
        tempi = open("Misure_temporali.txt", "a")
        tempi.write("\n\n\nFold numero %d" %tss_iteration)
        tempi.close()

        start_fit_ben_ensemble = time.process_time()
        training_features_ben=scaler1.fit_transform(training_features_ben)
        
        for i, (cluster_number, cluster_elements) in enumerate(clusters['Base'].items()):
            file_model = './model/'+device+'/model'+str(tss_iteration)+'/Ensemble'+str(i)
            if not os.path.isdir(file_model):
                Ensemble[i].fit(training_features_ben[:,cluster_elements], training_features_ben[:,cluster_elements], epochs=1)
                Ensemble[i].save(file_model)
            else:
                Ensemble[i]= tf.keras.models.load_model(file_model)
                    
        score=np.zeros((training_features_ben.shape[0],n_autoencoder))
      

        for j, (cluster_number, cluster_elements) in enumerate(clusters['Base'].items()):
            pred=Ensemble[j].predict(training_features_ben[:,cluster_elements])
            for i in range(training_features_ben.shape[0]):
                score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features_ben[i,cluster_elements]))

        time_fit_ensemble= time.process_time()-start_fit_ben_ensemble
        tempi = open("Misure_temporali.txt", "a")
        tempi.write("\n  Tempo di addestramento livello ensemble: %f secondi" %time_fit_ensemble)
        tempi.close()


        scaler2=MinMaxScaler(feature_range=(0,1))
        score=scaler2.fit_transform(score)

            #Training e generazione score Output layer
        print(training_features_ben.shape[0])
        update_size = int(np.ceil(training_features_ben.shape[0] / 5))
        test_features=scaler1.transform(test_features)
        training_features_mal=scaler1.transform(training_features_mal)

        for t in range(int(np.ceil(training_features_ben.shape[0] / update_size))):
            start_fit_out = time.process_time()
            file_model_out='./model/'+device+'/model'+str(tss_iteration)+'/Output'+str(t)
            if not os.path.isdir(file_model_out):
                Output.fit(score[t * update_size:(t + 1) * update_size,:],score[t * update_size:(t + 1) * update_size,:],epochs=1)
                if t == (int(np.ceil(training_features_ben.shape[0] / update_size))-1):
                    temp = ((training_features_ben.shape[0] / update_size) - int(training_features_ben.shape[0] / update_size))*update_size
                    if temp==0.0:
                        dim = update_size
                    else:
                        dim = temp
                        if dim-int(dim)>=0.5:
                            dim= int(np.ceil(dim))
                        else:
                            dim= int(np.floor(dim))
                else:
                    dim = update_size
                
                print("Dim= ",dim)
                training_labels_ben[t*dim:(t+1)*dim,3]=np.ones(dim)
                Output.save(file_model_out)
            else:
                Output=tf.keras.models.load_model(file_model_out)
            RMSE=np.zeros(score.shape[0])
            pred=Output.predict(score)

            for i in range(score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score[i]))

            time_fit_output= time.process_time()-start_fit_out
            tempi = open("Misure_temporali.txt", "a")
            tempi.write("\n\n  Iterazione %d:" %t)
            tempi.write("\n   Tempo di addestramento livello output: %f secondi" %time_fit_output)
            tempi.close()

            StampaValoriTrain(device, tss_iteration, train_index_ben, training_labels_ben, RMSE,0,t)




            #PREDIZIONE: PARTE MALEVOLA
            #tarin_mal_path = './SKF/'+device+'/Train_mal/SKF'+str(tss_iteration)
            #if not os.path.isdir(tarin_mal_path):
            start_mal = time.process_time()
            score_mal=np.zeros((training_features_mal.shape[0],n_autoencoder))
      
            #Generazione score Ensemble layer. i itera sulle entries del dataset, j sugli autoencoder/cluster
            for j, (cluster_number, cluster_elements) in enumerate(clusters['Base'].items()):
                pred=Ensemble[j].predict(training_features_mal[:,cluster_elements])
                for i in range(training_features_mal.shape[0]):
                    score_mal[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features_mal[i,cluster_elements]))
      
      
            score_mal=scaler2.transform(score_mal)

            #Training e generazione score Output layer
            RMSE=np.zeros(score_mal.shape[0])
            pred=Output.predict(score_mal)

            for i in range(score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score_mal[i]))


            time_mal= time.process_time()-start_mal
            tempi = open("Misure_temporali.txt", "a")
            tempi.write("\n   Tempo di predizione parte malevola: %f secondi" %time_mal)
            tempi.close()

            StampaValoriTrain(device, tss_iteration, train_index_mal, training_labels_mal, RMSE,1,t)



            # FASE DI TESTING TSS
            start_test = time.process_time()
            test_score=np.zeros((test_features.shape[0],n_autoencoder))
            #test_mal_path = './SKF/'+device+'/Test/SKF'+str(tss_iteration)
            #if not os.path.isdir(test_mal_path):

            for j, (cluster_number, cluster_elements) in enumerate(clusters['Base'].items()):
                pred=Ensemble[j].predict(test_features[:,cluster_elements])
                for i in range(test_features.shape[0]):
                    test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,cluster_elements]))
      
            test_score=scaler2.transform(test_score)
            RMSE=np.zeros(test_score.shape[0])
            pred=Output.predict(test_score)

            for i in range(test_score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
           
            time_test= time.process_time()-start_test
            tempi = open("Misure_temporali.txt", "a")
            tempi.write("   Tempo di predizione test: %f secondi" %time_test)
            tempi.close()

            StampaValoriTest(device, tss_iteration, test_index, test_labels, RMSE,t)

        tss_iteration = tss_iteration+1
