import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import seaborn as sns
sns.set()
import ipywidgets

from scipy import signal
from intervals import FloatInterval

def spectrogram_by_eeg_bandwidth(sig, sample_rate=125, timeIntervalSec=30):
    slow = FloatInterval.from_string('[0.5, 2.0)')
    delta = FloatInterval.from_string('[2, 4.0)')
    theta = FloatInterval.from_string('[4.0, 8.0)')
    alpha = FloatInterval.from_string('[8.0, 16.0)')
    beta = FloatInterval.from_string('[16.0, 32.0)')
    gamma = FloatInterval.from_string('[32.0, 100.0)')
    ##above100Hz = FloatInterval.from_string('[100.0,)')
    # no signal above100Hz
    nperseg = int(sample_rate * timeIntervalSec)
    noverlap = None #will take default value
    freqs, times, spec = signal.spectrogram(sig,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=sample_rate*timeIntervalSec,
                                    noverlap=noverlap,
                                    detrend=False)
    #Edelta, Etheta, Ealpha, Ebeta, Egamma, Eabove100Hz = np.zeros(6)
    result = pd.DataFrame(np.zeros(spec.shape[1]*6).reshape((6, spec.shape[1])), index=['Slow', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
    #
    for i in range(0, spec.shape[0]):
        for j in range(0, spec.shape[1]):
            if freqs[i] in slow:
                result.loc["Slow", j]= result.loc["Slow", j] + spec[i, j]
            if freqs[i] in delta:
                result.loc["Delta", j]= result.loc["Delta", j] + spec[i, j]
            elif freqs[i] in theta:
                result.loc["Theta", j]= result.loc["Theta", j] + spec[i, j]
            elif freqs[i] in alpha:
                result.loc["Alpha", j]= result.loc["Alpha", j] + spec[i, j]
            elif freqs[i] in beta:
                result.loc["Beta", j]= result.loc["Beta", j] + spec[i, j]
            elif freqs[i] in gamma:
                result.loc["Gamma", j]= result.loc["Gamma", j] + spec[i, j]
#            else: we are filtering out frequencies lower than 0.5 Hz
#               print("error at cell " + str((i, j)))
    return result
    
def generate_columns_names(L=['Slow', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], n=15):
    r = "["
    for i in L:
        for j in range(0,n):
            if (i == L[-1]) and (j==n-1):
                r = r + "'" + str(i)+ str(j) + "']"
            else:
                r = r + "'" + str(i)+ str(j) + "',"
    return eval(r)


def make_eeg_spectogram_dataframe(eeg, timeIntervalSec, columnsName=['Slow', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
    columns = generate_columns_names(n = 30//timeIntervalSec, L=columnsName)
    df = pd.DataFrame(columns = columns)
    for i in range(0, eeg.shape[0]):
        spec = spectrogram_by_eeg_bandwidth(eeg.iloc[i,:], 
            sample_rate=125, timeIntervalSec=timeIntervalSec)
        t = spec.values.reshape(1, spec.shape[0]*spec.shape[1])
        #df.loc[i] = [j for j in t[0]]
        df.loc[i] = t[0]
    return df   

def build_spectrogram_eeg_features(h5filename, timeIntervalSec, dataPath="C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\raw\\"):
    h5file= dataPath + h5filename
    h5 = h5py.File(h5file, "r")
    eeg_1 = pd.DataFrame(h5['eeg_1'][:])
    eeg_2 = pd.DataFrame(h5['eeg_2'][:])
    eeg_3 = pd.DataFrame(h5['eeg_3'][:])
    eeg_4 = pd.DataFrame(h5['eeg_4'][:])
    
    print("start FTT on eeg_1")
    eeg1 = make_eeg_spectogram_dataframe(eeg_1, timeIntervalSec, 
                                         columnsName=['eeg1_Slow', 'eeg1_Delta', 'eeg1_Theta', 'eeg1_Alpha', 'eeg1_Beta', 'eeg1_Gamma'])
    print("start FTT on eeg_2")
    eeg2 = make_eeg_spectogram_dataframe(eeg_2, timeIntervalSec, 
                                         columnsName=['eeg2_Slow','eeg2_Delta', 'eeg2_Theta', 'eeg2_Alpha', 'eeg2_Beta', 'eeg2_Gamma'])
    print("start FTT on eeg_3")
    eeg3 = make_eeg_spectogram_dataframe(eeg_3, timeIntervalSec, 
                                         columnsName=['eeg3_Slow','eeg3_Delta', 'eeg3_Theta', 'eeg3_Alpha', 'eeg3_Beta', 'eeg3_Gamma'])
    print("start FTT on eeg_4")
    eeg4 = make_eeg_spectogram_dataframe(eeg_4, timeIntervalSec, 
                                         columnsName=['eeg4_Slow','eeg4_Delta', 'eeg4_Theta', 'eeg4_Alpha', 'eeg4_Beta', 'eeg4_Gamma'])
    
    
    data = pd.concat([eeg1, eeg2, eeg3, eeg4], axis=1, sort=False)
    return data

#sampling rate = time interval 

data_test = build_spectrogram_eeg_features("test.h5", 30)
data_test.to_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\spectrogram_eeg_features30_more_Test.xlsx')

data_train = build_spectrogram_eeg_features("train.h5", 30)
data_train.to_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\spectrogram_eeg_features30_more_Train.xlsx')


