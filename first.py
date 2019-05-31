import h5py
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
 
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import seaborn as sns
sns.set()



#Datasets description: 43830 train samples for 20592 test samples

trainOutput = pd.read_csv("C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\challenge_fichier_de_sortie_dentrainement_classification_en_stade_de_sommeil_a_laide_de_signaux_mesures_par_le_bandeau_dreem.csv", sep=";")

#sleep classification follows the AASM recommendations and was labeled by a single expert.
# 0 : Wake
# 1 : N1 (light sleep)
# 2 : N2
# 3 : N3 (deep sleep)
# 4 : REM (paradoxal sleep)


filetrain= "C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\train.h5"
filetest= "C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\test.h5"
#keys: eeg_1, eeg_2, eeg_3, eeg_4, po_ir, po_r, accelerometer_x, accelerometer_y, accelerometer_z . Any of those datasets can then be accessed using :

h5 = h5py.File(filetrain, "r")

#4 EEG channels sampled at 125Hz (4 x 125 x 30 = 15000 size per sample)
# for each eeg, [43830 rows x 3750 columns]: each sample is 30 secs with 125 Hz
reeg1 = pd.DataFrame(h5['eeg_1'][:])

reeg1 = eeg1.copy()
reeg1["Y"] = trainOutput.copy()["label"].astype('category')

# repartition des phases
phases = reeg1.groupby(["Y"]).groups
for i in phases.keys():
    print(str(i) + " : " + str(phases[i].size))

#0 : 4939
#1 : 1359
#2 : 16139
#3 : 13780
#4 : 7613    
    
    
    
sns.distplot(reeg1.loc[:,"Y"]);
sns.distplot(reeg1.iloc[1])

sns.kdeplot(reeg1.iloc[1], shade=True)

def vardensity(df):
    for i in range(0, 10):
        sns.kdeplot(df.iloc[i], shade=True)
        plt.figure()


def getsample(df):
    train, validate = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])
    return [train, validate]
    
#43830 rows
#35064 rows
#8766 rows

# plot eeg1 by subject with reeg1


palette = plt.get_cmap('Set1')
plt.plot(reeg1.iloc[1, 0:-1], color=palette(reeg1.iloc[1, -1]))
plt.legend(loc=2, ncol=2)

def plot_eeg1(df, r):
    for i in range(0, r):
        plt.plot(df.iloc[i, 0:-1], color=palette(reeg1.iloc[i, -1]))
        plt.title("subject: " + str(i) +" phase: " + str(reeg1.iloc[i, -1]), loc='left', fontsize=18, fontweight=0, color='blue')
        plt.figure()


####
#acp step 1
meeg1 = h5['eeg_1'][:]
meeg1_centered = meeg1 - meeg1.mean(axis=0)
U, s, V = np.linalg.svd(meeg1_centered) 
c1 = V.T[:, 0] 
c2 = V.T[:, 1]

pca = PCA(n_components = 2, svd_solver= 'full', whiten = True) 
meeg1_2D = pca.fit_transform(meeg1)
print(pca.explained_variance_ratio_)


pca = PCA(svd_solver= 'full', whiten = True)
pca.fit(meeg1)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
#d = 34
cumsum_2d = np.cumsum(pca.explained_variance_ratio_[0:2*d])
plt.plot(cumsum_2d)
 
 
 pca34 = PCA(n_components = 34, svd_solver= 'full', whiten = True) 
 meeg1_34 = pca34.fit_transform(meeg1)

reeg1_34 = pd.DataFrame(meeg1_34)

def plot_eeg1_34(df, r):
    for i in range(0, r):
        plt.plot(df.iloc[i,:], color="blue")
        plt.title("subject: " + str(i), loc='left', fontsize=18, fontweight=0, color='blue')
        plt.figure()

plot_eeg1_34(reeg1_34, 10)

kmeans34 = KMeans(n_clusters=5)
#kmeans34.fit(reeg1_34)
#y34_kmeans = kmeans34.predict(reeg1_34)
clusters34 = kmeans34.fit_predict(reeg1_34)

Y = trainOutput.copy()["label"].astype('category')

#matching each learned cluster label with the true labels found in them:
labels34 = np.zeros_like(clusters)
for i in range(5):
    mask = (clusters34 == i)
    labels34[mask] = mode(Y[mask])[0]

accuracy_score(Y, labels34)
#0.37
#Cohen_Kappa_score: 0.00018135659926832304


#from sklearn.metrics import accuracy_score
#accuracy_score(digits.target, labels)



# manifold learning better with non linear relationships
# https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html