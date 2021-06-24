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
from sklearn.metrics import recall_score
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
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


class Algoritmi(Enum):
    Alg1 = 0
    Alg2 = 1
    Alg3 = 2
    NATD = 3
    SAT1 = 4
    SAT10 = 5
    SAT50 = 6
    SATall = 7


def compute_roc(device):
    dataset = dict()
    fig_path = "./Grafici/ROC/"
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive','yellow','tab:brown','black']
    linestyle = ['--']
    markers = ["o","D","v","p","s","X","x", "|"]
        

    tprs = []
    mean_fpr = np.linspace(0,1,10000)

    for tss_iteration in range(10):

        path = './SKF/'+device.name+'/Test/SKF'+str(tss_iteration)

            
        dataset = pd.read_parquet(path+'/SKF4.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for j in range(0,len(dataset[:,3])):
            if dataset[j,3]!=0:
                dataset[j,3]=1

            
        fpr,tpr,thresholds= metrics.roc_curve(dataset[:,3],dataset[:,4])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.plot(mean_fpr, mean_tpr, 'k:', alpha=.3,ls= linestyle[0], linewidth = 2)

    thr = dict()
    thr = pd.read_csv('./Soglie/Thr'+str(device.value)+'.csv', delimiter=',')
    thr = pd.DataFrame(thr)
    thr = thr.to_numpy().astype('float32')
    for i in range(8):
        if i==5: continue
        indices = np.where(thresholds<=thr[i,3])
        #Le soglie sono ordinate in modo decrescente
        index = np.min(indices)
        #print(fpr[index])
        #print(tpr[index])
        plt.plot(fpr[index],tpr[index], marker=markers[i], color=colors[i], label=Algoritmi(i).name)
        plt.errorbar(fpr[index],tpr[index], xerr=thr[i,4], color=colors[i])
        plt.errorbar(fpr[index],tpr[index], xerr=-thr[i,4], color=colors[i])

    plt.plot([1e-2, 1e-2], [0, 10], 'k:', alpha = .3)

    plt.xlim([0, 0.5])
    plt.ylim([0, 1.05])
    plt.yticks(np.arange(0, 1.05, .1))
    #plt.xticks(np.arange(0.01, 0.5, .001))
    plt.xscale('symlog', linthreshx =1e-3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - '+device.name.replace('_',' '))
    plt.legend(loc='lower right', fontsize=12)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()






#CREA UN ISTOGRAMMA PER OGNI DISPOSITIVO RIPORTANDO I VALORI DI MEDIA E DEVIAZIONE STANDARD DELL'F1-SCORE AL VARIARE DELL'ALGORITMO
def compute_f1(device):
    dataset = dict()
    fig_path = "./Grafici/f1_algoritmi/"
    f1_score = np.zeros(10)
    a = 0
    

    if device.value!=2 and device.value!=6:

        
        f1_alg_mean = np.zeros(8)
        f1_alg_std = np.zeros(8)
        for alg in ['algoritmo1','algoritmo2','algoritmo3','naive_anomaly_threshold_with_decay','stochastic_anomaly_threshold_1','stochastic_anomaly_threshold_50','stochastic_anomaly_threshold_all','Upperbound']:
            
            if alg != 'Upperbound':
                for tss_iteration in range(10):
                    path = './Risultati/'+device.name+'/'+alg

                    dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
                    dataset = pd.DataFrame(dataset)
                    dataset = dataset.to_numpy().astype('float32') 
            
                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,3]!=0:
                            dataset[j,3]=1

                    f1_score[tss_iteration]=metrics.f1_score(dataset[:,3],dataset[:,1],average='macro')

                f1_alg_mean[a] = round(f1_score.mean(),2)
                f1_alg_std[a] = round(f1_score.std(),2)
                a = a+1
            else:
                for tss_iteration in range(10):
                    path = './SKF/'+device.name+'/Test/SKF'+str(tss_iteration)

                    dataset = pd.read_parquet(path+'/SKF4.parquet')
                    dataset = pd.DataFrame(dataset)
                    dataset = dataset.to_numpy().astype('float32')


                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,3]!=0:
                            dataset[j,3]=1
 

                    fpr,tpr,thresholds= metrics.roc_curve(dataset[:,3],dataset[:,4])
                    indices = np.where(fpr>=0.01)
                    index = np.min(indices)
                    soglia = thresholds[index]

                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,4] <= soglia:
                            dataset[j,1] = 0
                        else:
                            dataset[j,1] = 1

                    f1_score[tss_iteration]=metrics.f1_score(dataset[:,3],dataset[:,1],average='macro')

                f1_alg_mean[a] = round(f1_score.mean(),2)
                f1_alg_std[a] = round(f1_score.std(),2)

    
        labels = ['Alg1', 'Alg2', 'Alg3', 'NATD', 'SAT1', 'SAT50', 'SATall', 'FPR1%']

        x = np.arange(len(labels)) 
        width = 0.6 

        fig, ax = plt.subplots(figsize=(7,7))
        rects1 = ax.bar(x, f1_alg_mean, yerr=f1_alg_std, width=width, capsize=5)
        #rects2 = ax.errorbar(x,f1_alg_mean, yerr=f1_alg_std, ecolor='black', label='std')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        #ax.legend(loc='lower right', fontsize=12)
    
        plt.ylim((0,1.1))
        plt.ylabel("F1-score", fontsize=12)
        plt.title(device.name.replace('_',' '), fontsize=12)

        ax.bar_label(rects1, padding=3, label_type='center', fontsize=12)
        #ax.bar_label(rects2, padding=3, label_type='center', fontsize=12)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fig_path+device.name+".png")
        plt.close()

    else:
        f1_alg_mean = np.zeros(5)
        f1_alg_std = np.zeros(5)
        for alg in ['algoritmo1','algoritmo2','algoritmo3','naive_anomaly_threshold_with_decay', 'Upperbound']:

            if alg != 'Upperbound':
                for tss_iteration in range(10):
                    path = './Risultati/'+device.name+'/'+alg


                    dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
                    dataset = pd.DataFrame(dataset)
                    dataset = dataset.to_numpy().astype('float32') 
         
                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,3]!=0:
                            dataset[j,3]=1

                    f1_score[tss_iteration]=metrics.f1_score(dataset[:,3],dataset[:,1],average='macro')

                f1_alg_mean[a] = round(f1_score.mean(),2)
                f1_alg_std[a] = round(f1_score.std(),2)
                a = a+1
            else:
                for tss_iteration in range(10):
                    path = './SKF/'+device.name+'/Test/SKF'+str(tss_iteration)

                    dataset = pd.read_parquet(path+'/SKF4.parquet')
                    dataset = pd.DataFrame(dataset)
                    dataset = dataset.to_numpy().astype('float32') 

                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,3]!=0:
                            dataset[j,3]=1

                    fpr,tpr,thresholds= metrics.roc_curve(dataset[:,3],dataset[:,4])
                    indices = np.where(fpr>=0.01)
                    index = np.min(indices)
                    soglia = thresholds[index]

                    for j in range(0,len(dataset[:,3])):
                        if dataset[j,4] <= soglia:
                            dataset[j,1] = 0
                        else:
                            dataset[j,1] = 1

                    f1_score[tss_iteration]=metrics.f1_score(dataset[:,3],dataset[:,1],average='macro')

                f1_alg_mean[a] = round(f1_score.mean(),2)
                f1_alg_std[a] = round(f1_score.std(),2)

    
        labels = ['Alg1', 'Alg2', 'Alg3', 'NATD', 'FPR1%']

        x = np.arange(len(labels)) 
        width = 0.6 

        fig, ax = plt.subplots(figsize=(7,7))
        rects1 = ax.bar(x, f1_alg_mean, yerr=f1_alg_std, width=width, capsize=5)
        #rects2 = ax.bar(x, f1_alg_std, width, color='orange', bottom=f1_alg_mean, label='std')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        #ax.legend(loc='lower right', fontsize=12)
    
        plt.ylim((0,1.1))
        plt.ylabel("F1-score", fontsize=12)
        plt.title(device.name.replace('_',' '), fontsize=12)

        ax.bar_label(rects1, padding=3, label_type='center', fontsize=12)
        #ax.bar_label(rects2, padding=3, label_type='center', fontsize=12)

        if not os.path.isdir(fig_path):
           os.makedirs(fig_path)

        plt.savefig(fig_path+device.name+".png")
        plt.close()




#CONSIDERANDO L'ALGORITMO OTTIMALE PER OGNI DISPOSITIVO SI VERIFICA QUALE ATTACCO VIENE RILEVATO MEGLIO. VIENE RIPORTATA LA DETECTION RATE 
def detectionrate_attack_opt(device):

    if device.name == Device(0).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(1).name: alg = "algoritmo2"
    elif device.name == Device(2).name: alg = "algoritmo1"
    elif device.name == Device(3).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(4).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(5).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(6).name: alg = "algoritmo3"
    elif device.name == Device(7).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(8).name: alg = "algoritmo3"


    fig_path = "./Grafici/f1_attack_opt/"
    tot_oss = np.zeros(11)
    tot_pos = np.zeros(11)


    for tss_iteration in range(10):

        path = './Risultati/'+device.name+'/'+alg

        t=0
        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')


        for attack in Attack:
            n_malign = 0
            positive = 0
            if device.value == 2 or device.value == 6:
                if t>=6:
                    continue

            for j in range(0, len(dataset[:,1])):
                if dataset[j,3]==attack.value:
                    n_malign = n_malign+1
                    if dataset[j,3]!=0:
                        if dataset[j,1]==1:
                            positive = positive+1
                    else:
                        if dataset[j,1]==0:
                            positive = positive+1
                    
            tot_oss[t] = tot_oss[t] + n_malign
            tot_pos[t] = tot_pos[t] + positive
            t=t+1


    if device.value == 2 or device.value == 6:
        labels = ['Ben', 'combo', 'junk', 'scan', 'tcp', 'udp']
        percentage = np.zeros(6)
        for i in range(6):
            percentage[i] = round(tot_pos[i]/tot_oss[i],2)
    else:
        labels = ['Ben', 'combo', 'junk', 'b_scan', 'tcp', 'b_udp', 'ack', 'm_scan', 'syn', 'm_udp', 'udpplain']
        percentage = np.zeros(11)
        for i in range(11):
            percentage[i] = round(tot_pos[i]/tot_oss[i],2)


    alg_str_fix = ""
    if alg == 'algoritmo1': alg_str_fix = "Alg1"
    elif alg == 'algoritmo2': alg_str_fix = "Alg2"
    elif alg == 'algoritmo3': alg_str_fix = "Alg3"
    elif alg == 'stochastic_anomaly_threshold_50': alg_str_fix = "SAT50"



    x = np.arange(len(labels)) 
    width = 0.6 

    fig, ax = plt.subplots(figsize=(7,4))
    rects1 = ax.bar(x, percentage, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=20)
    
    plt.ylim((0,1))
    plt.ylabel("Detection Rate", fontsize=10)
    plt.title(device.name.replace('_',' ')+' - '+alg_str_fix, fontsize=12)

    ax.bar_label(rects1, padding=3, label_type='center', fontsize=12)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()







#CON S.A.T. CON TUTTO IL TRAFFICO MIRAI, CALCOLA LA PERCENTUALE DI PACCHETTI BASHLITE RILEVATI
def detection_bashlite(device):
    
    fig_path = "./Grafici/Bashlite/bashlite_all/"
    dataset = dict()
    n_bashlite = 0
    positive = 0
    negative = 0


    for tss_iteration in range(0,10):
        path = './Risultati/'+device.name+'/stochastic_anomaly_threshold_all'

        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for j in range(0, len(dataset[:,1])):
            if dataset[j,3]>=1 and dataset[j,3]<=5:
                n_bashlite = n_bashlite+1
                if dataset[j,1]==1:
                    positive = positive+1
                else:
                    negative = negative+1
      

    labels = ['Rilevati', 'Non rilevati']
    percentage = [(positive*100)/n_bashlite, (negative*100)/n_bashlite]

    fig1, ax = plt.subplots(figsize=(4,4))
    
    ax.pie(percentage, colors=['tab:blue','tab:orange'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.legend(loc='upper right', labels=labels, fontsize=12)
    ax.axis('equal')
    plt.title(device.name.replace('_',' ')+' (a)', fontsize=11)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()


#CON S.A.T. CON TUTTO IL TRAFFICO MIRAI, CALCOLA LA PERCENTUALE DI PACCHETTI BENIGNI RILEVATI
def detection_benign(device):
    
    fig_path = "./Grafici/Bashlite/benign_all/"
    dataset = dict()
    n_benign = 0
    positive = 0
    negative = 0


    for tss_iteration in range(0,10):
        path = './Risultati/'+device.name+'/stochastic_anomaly_threshold_all'

        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for j in range(0, len(dataset[:,1])):
            if dataset[j,3]==0:
                n_benign = n_benign+1
                if dataset[j,1]==0:
                    positive = positive+1
                else:
                    negative = negative+1
      

    labels = ['Rilevati', 'Non rilevati']
    percentage = [(positive*100)/n_benign, (negative*100)/n_benign]

    fig1, ax = plt.subplots(figsize=(4,4))
    
    ax.pie(percentage, colors=['tab:green','tab:red'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.legend(loc='upper right', labels=labels, fontsize=12)
    ax.axis('equal')
    plt.title(device.name.replace('_',' ')+' (b)', fontsize=11)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()




#CON S.A.T CON TUTTO IL TRAFFICO MIRAI SI CALCOLA LA PERCENTUALE DI PACCHETTI RILEVATI DELLE DIVERSE TIPOLOGIE DI ATTACCO
def bashlite_attack_percentage(device):

    fig_path = "./Grafici/Bashlite/bashlite_attacks/"
    dataset = dict()
    percentage = np.zeros(5)
    tot_oss = np.zeros((10,5))
    tot_pos = np.zeros((10,5))
    n_positivi = np.zeros(5)
    n_osservazioni = np.zeros(5)


    for tss_iteration in range(10):
        path = './Risultati/'+device.name+'/stochastic_anomaly_threshold_all'

        k=0
        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for attack in Attack:
            oss = 0
            positive = 0
            if attack.value==0 or attack.value>5: continue
            #tipologia = [j for j in dataset[:,3] if j==Attack[attack.name].value]
            for j in range(0,len(dataset[:,1])):
                if dataset[j,3]==attack.value:
                    oss = oss+1
                    if dataset[j,1]==1:
                        positive = positive+1
            
            tot_oss[tss_iteration][k] = oss
            tot_pos[tss_iteration][k] = positive
            k=k+1


    for tss_iteration in range(10):
        for j in range(5):
            n_osservazioni[j] = n_osservazioni[j]+tot_oss[tss_iteration][j]
            n_positivi[j] = n_positivi[j]+tot_pos[tss_iteration][j]


    percentage = [round((n_positivi[0]*100)/n_osservazioni[0],2), round((n_positivi[1]*100)/n_osservazioni[1],2), round((n_positivi[2]*100)/n_osservazioni[2],2), round((n_positivi[3]*100)/n_osservazioni[3],2), round((n_positivi[4]*100)/n_osservazioni[4],2)]
    labels = ['combo', 'junk', 'scan', 'tcp', 'udp']
    
    plt.figure(figsize=(7,6))
    r = plt.bar(labels, percentage, color=['navy','orange','green','darkred','goldenrod'])
    plt.bar_label(r, padding=3, label_type='center', fontsize=12, color='w')

    plt.ylim(0,100)
    plt.ylabel("Detection Rate (%)", fontsize=12)
    plt.title(device.name.replace('_',' '), fontsize=12)


    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()



#CON GLI ALGORITMO OTTIMALI SI CALCOLANO LE PERCENTUALI DEI PACCHETTI BASHLITE RILEVATI
def bashlite_opt(device):

    if device.name == Device(0).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(1).name: alg = "algoritmo2"
    elif device.name == Device(2).name: alg = "algoritmo1"
    elif device.name == Device(3).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(4).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(5).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(6).name: alg = "algoritmo3"
    elif device.name == Device(7).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(8).name: alg = "algoritmo3"


    fig_path = "./Grafici/Bashlite/bashlite_all_opt/"
    dataset = dict()
    report = dict()
    n_bashlite = 0
    positive = 0
    negative = 0

    for tss_iteration in range(10):

        path = './Risultati/'+device.name+'/'+alg

        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for j in range(0, len(dataset[:,1])):
            if dataset[j,3]>=1 and dataset[j,3]<=5:
                n_bashlite = n_bashlite+1
                if dataset[j,1]==1:
                    positive = positive+1
                else:
                    negative = negative+1
      

    labels = ['Rilevati', 'Non rilevati']
    percentage = [(positive*100)/n_bashlite, (negative*100)/n_bashlite]

    fig1, ax = plt.subplots(figsize=(4,4))
    
    ax.pie(percentage, colors=['tab:blue','tab:orange'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.legend(loc='upper right', labels=labels, fontsize=12)
    ax.axis('equal')
    plt.title(device.name.replace('_',' ')+' (a)', fontsize=11)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()


#PERCENTUALE DI PACCHETTI BENIGNI RILEVATI E NON RILEVATI UTILIZZANDO L'ALGORITMO OTTIMO PER OGNI DISPOSITIVO
def detection_benign_opt(device):
    
    if device.name == Device(0).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(1).name: alg = "algoritmo2"
    elif device.name == Device(2).name: alg = "algoritmo1"
    elif device.name == Device(3).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(4).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(5).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(6).name: alg = "algoritmo3"
    elif device.name == Device(7).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(8).name: alg = "algoritmo3"


    fig_path = "./Grafici/Bashlite/benign_all_opt/"
    dataset = dict()
    n_benign = 0
    positive = 0
    negative = 0


    for tss_iteration in range(0,10):
        path = './Risultati/'+device.name+'/'+alg

        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for j in range(0, len(dataset[:,1])):
            if dataset[j,3]==0:
                n_benign = n_benign+1
                if dataset[j,1]==0:
                    positive = positive+1
                else:
                    negative = negative+1
      

    labels = ['Rilevati', 'Non rilevati']
    percentage = [(positive*100)/n_benign, (negative*100)/n_benign]

    fig1, ax = plt.subplots(figsize=(4,4))
    
    ax.pie(percentage, colors=['tab:green','tab:red'], autopct='%1.1f%%', shadow=True, startangle=0)
    ax.legend(loc='upper right', labels=labels, fontsize=12)
    ax.axis('equal')
    plt.title(device.name.replace('_',' ')+' (b)', fontsize=11)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()





#PERCENTUALE DI PACCHETTI RELATIVI AGLI ATTACCHI BASHLITE RILEVATI UTILIZZANDO L'ALGORITMO OTTIMO PER OGNI DISPOSITIVO
def bashlite_attack_percentage_opt(device):


    if device.name == Device(0).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(1).name: alg = "algoritmo2"
    elif device.name == Device(2).name: alg = "algoritmo1"
    elif device.name == Device(3).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(4).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(5).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(6).name: alg = "algoritmo3"
    elif device.name == Device(7).name: alg = "stochastic_anomaly_threshold_50"
    elif device.name == Device(8).name: alg = "algoritmo3"


    fig_path = "./Grafici/Bashlite/bashlite_attacks_opt/"
    dataset = dict()
    percentage = np.zeros(5)
    tot_oss = np.zeros((10,5))
    tot_pos = np.zeros((10,5))
    n_positivi = np.zeros(5)
    n_osservazioni = np.zeros(5)


    for tss_iteration in range(10):
        path = './Risultati/'+device.name+'/'+alg

        k=0
        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for attack in Attack:
            positive = 0
            oss=0
            if attack.value==0 or attack.value>5: continue
            for j in range(0,len(dataset[:,1])):
                if dataset[j,3]==attack.value:
                    oss = oss+1
                    if int(dataset[j,1])==1:
                        positive = positive+1
            
            tot_oss[tss_iteration][k] = oss
            tot_pos[tss_iteration][k] = positive
            k=k+1


    for tss_iteration in range(10):
        for j in range(5):
            n_osservazioni[j] = n_osservazioni[j]+tot_oss[tss_iteration][j]
            n_positivi[j] = n_positivi[j]+tot_pos[tss_iteration][j]


    percentage = [round((n_positivi[0]*100)/n_osservazioni[0],2), round((n_positivi[1]*100)/n_osservazioni[1],2), round((n_positivi[2]*100)/n_osservazioni[2],2), round((n_positivi[3]*100)/n_osservazioni[3],2), round((n_positivi[4]*100)/n_osservazioni[4],2)]
    labels = ['combo', 'junk', 'scan', 'tcp', 'udp']
    
    plt.figure(figsize=(7,6))
    r = plt.bar(labels, percentage, color=['navy','orange','green','darkred','goldenrod'])
    plt.bar_label(r, padding=3, label_type='center', fontsize=12, color='w')

    plt.ylim(0,100)
    plt.ylabel("Detection Rate (%)", fontsize=12)
    plt.title(device.name.replace('_',' '), fontsize=12)


    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()




#CON S.A.T. CON 50 PACCHETTI DI SCAN_MIRAI SI VERIFICA QUALE TIPOLOGIA DI ATTACCO VIENE RILEVATA MEGLIO
def attacks_detectionrate_sat(device):
    
    
    fig_path = "./Grafici/f1_attacks_sat50/"
    dataset = dict()
    tot_oss = np.zeros(11)
    tot_pos = np.zeros(11)


    for tss_iteration in range(10):

        path = './Risultati/'+device.name+'/stochastic_anomaly_threshold_50'

        t=0
        dataset = pd.read_parquet(path+'/SKF'+str(tss_iteration)+'.parquet')
        dataset = pd.DataFrame(dataset)
        dataset = dataset.to_numpy().astype('float32')

        for attack in Attack:
            n_malign = 0
            positive = 0
            if device.value == 2 or device.value == 6:
                if t>=6:
                    continue

            for j in range(0, len(dataset[:,1])):
                if dataset[j,3]==attack.value:
                    n_malign = n_malign+1
                    if dataset[j,3]!=0:
                        if dataset[j,1]==1:
                            positive = positive+1
                    else:
                        if dataset[j,1]==0:
                            positive = positive+1
                    
            tot_oss[t] = tot_oss[t] + n_malign
            tot_pos[t] = tot_pos[t] + positive
            t=t+1

    
    if device.value == 2 or device.value == 6:
        labels = ['Ben', 'combo', 'junk', 'scan', 'tcp', 'udp']
        percentage = np.zeros(6)
        for i in range(6):
            percentage[i] = round(tot_pos[i]/tot_oss[i],2)
    else:
        labels = ['Ben', 'combo', 'junk', 'b_scan', 'tcp', 'b_udp', 'ack', 'm_scan', 'syn', 'm_udp', 'udpplain']
        percentage = np.zeros(11)
        for i in range(11):
            percentage[i] = round(tot_pos[i]/tot_oss[i],2)


    x = np.arange(len(labels)) 
    width = 0.6 

    fig, ax = plt.subplots(figsize=(7,4))
    rects1 = ax.bar(x, percentage, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=20)
    
    plt.ylim((0,1))
    plt.ylabel("Detection Rate", fontsize=10)
    plt.title(device.name.replace('_',' '), fontsize=12)

    ax.bar_label(rects1, padding=3, label_type='center', fontsize=12)

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    plt.savefig(fig_path+device.name+".png")
    plt.close()







os.chdir('/home/francesco/Scrivania/Codice/')
dataset = dict()


for device in Device:

    #compute_f1(device)
    #detectionrate_attack_opt(device)
    

    #if device.value!=2 and device.value!=6:
    #    compute_roc(device)
        #detection_bashlite(device)
        #detection_benign(device)
        #bashlite_attack_percentage(device)
        #bashlite_opt(device)
        #detection_benign_opt(device)
        #bashlite_attack_percentage_opt(device)
        #attacks_detectionrate_sat(device)

    #if device.value==2 or device.value==6:
    #    compute_roc(device)

