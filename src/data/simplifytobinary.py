import h5py
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import cohen_kappa_score

import statsmodels.api as sm

from scipy.stats import mode

import random
 
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import seaborn as sns
sns.set()



#Datasets description: 43830 train samples for 20592 test samples

dataPath = "C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\raw\\"

trainOutput = pd.read_csv(dataPath + "challenge_fichier_de_sortie_dentrainement_classification_en_stade_de_sommeil_a_laide_de_signaux_mesures_par_le_bandeau_dreem.csv", sep=";")
Y = trainOutput["label"]

#sleep classification follows the AASM recommendations and was labeled by a single expert.
# 0 : Wake
# 1 : N1 (light sleep)
# 2 : N2
# 3 : N3 (deep sleep)
# 4 : REM (paradoxal sleep)



def phasemap(n):
    r ="error"
    if (n==0): 
        r = "Wake"
    elif (n==1):
        r = "N1 (Light sleep)"
    elif (n==2):
        r = "N2"
    elif (n==3):
        r = "N3 (Deep sleep)"
    elif (n==4):
        r = "REM (paradoxal sleep"
    return r


filetrain= dataPath + "train.h5"
filetest= dataPath + "test.h5"
#keys: eeg_1, eeg_2, eeg_3, eeg_4, po_ir, po_r, accelerometer_x, accelerometer_y, accelerometer_z . 
#Any of those datasets can then be accessed using :

h5 = h5py.File(filetrain, "r")
h5keys = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_ir', 'po_r', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z']

#4 EEG channels sampled at 125Hz (4 x 125 x 30 = 15000 size per sample)
# for each eeg, [43830 rows x 3750 columns]: each sample is 30 secs with 125 Hz
# 2 Pulse Oxymeter channels (red and infra-red) sampled at 50 Hz (2 x 50 x 30 = 3000 size per sample)
# 3 Accelerometer channels sampled at 50Hz (3 x 50 x 30 = 4500 size per sample)


#eeg1 = pd.DataFrame(h5['eeg_1'][:])


eeg_1 = pd.DataFrame(h5['eeg_1'][:])
eeg_2 = pd.DataFrame(h5['eeg_2'][:])
eeg_3 = pd.DataFrame(h5['eeg_3'][:])
eeg_4 = pd.DataFrame(h5['eeg_4'][:])
po_ir = pd.DataFrame(h5['po_ir'][:])
po_r = pd.DataFrame(h5['po_r'][:])
accelerometer_x = pd.DataFrame(h5['accelerometer_x'][:])
accelerometer_y = pd.DataFrame(h5['accelerometer_y'][:])
accelerometer_z = pd.DataFrame(h5['accelerometer_z'][:])


###########################################################################################################
# Viz by rows


palette = plt.get_cmap('Set1')
#plt.plot(eeg_1.iloc[1], color=palette(eeg_1.iloc[1, -1]))
plt.plot(eeg_1.iloc[1])
plt.legend(loc=2, ncol=2)

def plot_eeg1(df, r):
    for i in range(0, r):
        plt.plot(df.iloc[i, 0:-1], color=palette(reeg1.iloc[i, -1]))
        plt.title("subject: " + str(i) +" phase: " + str(reeg1.iloc[i, -1]), loc='left', fontsize=18, fontweight=0, color='blue')
        plt.figure()

def plot_row(r):
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots(figsize=(20,10))
    #plt.xlim(0,3750)
    #plt.ylim(-2,22)
    plt.subplot(4,3, 1)
    plt.plot(eeg_1.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 4)
    plt.plot(eeg_2.iloc[r], color=palette(Y[r]))    
    plt.subplot(4,3, 7)
    plt.plot(eeg_3.iloc[r], color=palette(Y[r]))    
    plt.subplot(4,3, 10)
    plt.plot(eeg_4.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 2)
    plt.plot(po_ir.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 5)
    plt.plot(po_r.iloc[r], color=palette(Y[r])) 
    plt.subplot(4,3, 3)
    plt.plot(accelerometer_x.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 6)
    plt.plot(accelerometer_y.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 9)
    plt.plot(accelerometer_y.iloc[r], color=palette(Y[r]))
    plt.suptitle(phasemap(Y[r]), fontsize=18, fontweight=5, color='black', style='italic', y=1.02)

    
def rowdensity(r):
# sur les lignes !
    palette = plt.get_cmap('Set1')
    fig, ax = plt.subplots(figsize=(20,10))
    #plt.xlim(0,3750)
    #plt.ylim(-2,22)
    plt.subplot(4,3, 1)
    sns.kdeplot(eeg_1.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 4)
    sns.kdeplot(eeg_2.iloc[r], color=palette(Y[r]))    
    plt.subplot(4,3, 7)
    sns.kdeplot(eeg_3.iloc[r], color=palette(Y[r]))    
    plt.subplot(4,3, 10)
    sns.kdeplot(eeg_4.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 2)
    sns.kdeplot(po_ir.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 5)
    sns.kdeplot(po_r.iloc[r], color=palette(Y[r])) 
    plt.subplot(4,3, 3)
    sns.kdeplot(accelerometer_x.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 6)
    sns.kdeplot(accelerometer_y.iloc[r], color=palette(Y[r]))
    plt.subplot(4,3, 9)
    sns.kdeplot(accelerometer_y.iloc[r], color=palette(Y[r]))
    plt.suptitle(phasemap(Y[r]), fontsize=18, fontweight=5, color='black', style='italic', y=1.02)
    
    

    
def plot_rowsamples(i):
    for i in range(0, i):
        r = randrange(0, 4382)
        plot_row(r)
        rowdensity(r)


    
        
reeg1 = eeg1.copy()
reeg1["Y"] = trainOutput.copy()["label"].astype('category')
    

plt.hist(Y, alpha=0.5, bins=[0, 1, 2, 3, 4, 5])

   

#viz by features


 
def plot_varsamplesdensity(i):
    for i in range(0, i):
        for k in h5keys:
            var = eval(k + "[random.choice(" + k + ".columns)]")
            sns.kdeplot(var, shade=True)
            plt.title(k, loc='left', fontsize=18, fontweight=3)
            plt.figure()

def plot_varsamplesdensity_phases(i):
    for i in range(0, i):
        for k in h5keys:
            df = eval(k)
            c = random.choice(df.columns)
            df["Y"] = Y
            wake    = df.loc[df.Y==0][c]
            N1      = df.loc[df.Y==1][c]
            N2      = df.loc[df.Y==2][c]
            N3      = df.loc[df.Y==3][c]
            REM     = df.loc[df.Y==4][c]
            fig, ax = plt.subplots(figsize=(30,10))
            plt.subplot(1,5, 1)
            sns.kdeplot(wake, shade=True)
            plt.subplot(1,5, 2)
            sns.kdeplot(N1, shade=True) 
            plt.subplot(1,5, 3)
            sns.kdeplot(N2, shade=True) 
            plt.subplot(1,5, 4)
            sns.kdeplot(N3, shade=True) 
            plt.subplot(1,5, 5)
            sns.kdeplot(REM, shade=True) 
            plt.title("Sensor: " + str(k) + " Var: " + str(c), loc='left', fontsize=18, fontweight=3)
            plt.suptitle("Sensor: " + str(k) + " Var: " + str(c), fontsize=18, fontweight=5, color='black', style='italic', y=1.02)
            plt.figure()            

            
            
            
# 0 : Wake
# 1 : N1 (light sleep)
# 2 : N2
# 3 : N3 (deep sleep)
# 4 : REM (paradoxal sleep)           
            

##################################################################################
# create features to capture distribution for each sensors
# mean, std, min, max, 25%, 50%, 75%




def build_data():
    filetrain= dataPath + "train.h5"
    h5 = h5py.File(filetrain, "r")
    h5keys = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_ir', 'po_r', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z']
    eeg_1 = pd.DataFrame(h5['eeg_1'][:])
    eeg_2 = pd.DataFrame(h5['eeg_2'][:])
    eeg_3 = pd.DataFrame(h5['eeg_3'][:])
    eeg_4 = pd.DataFrame(h5['eeg_4'][:])
    po_ir = pd.DataFrame(h5['po_ir'][:])
    po_r = pd.DataFrame(h5['po_r'][:])
    accelerometer_x = pd.DataFrame(h5['accelerometer_x'][:])
    accelerometer_y = pd.DataFrame(h5['accelerometer_y'][:])
    accelerometer_z = pd.DataFrame(h5['accelerometer_z'][:])
    
    data = pd.DataFrame()
    for k in h5keys:
        df = eval(k)
        data["mean_" + k] = df.mean(axis=1)
        data["std_" + k] = df.std(axis=1)
        data["min_" + k] = df.min(axis=1)
        data["max_" + k] = df.max(axis=1)
        data["25%_" + k] = df.quantile(0.25, axis=1)
        data["50%_" + k] = df.quantile(0.50, axis=1)
        data["75%_" + k] = df.quantile(0.75, axis=1)
        print(k)
    trainOutput = pd.read_csv(dataPath + "challenge_fichier_de_sortie_dentrainement_classification_en_stade_de_sommeil_a_laide_de_signaux_mesures_par_le_bandeau_dreem.csv", sep=";")
    data["Y"] = trainOutput["label"]
    return data

data = build_data()    
sns.countplot(x="Y", data=data)    

# repartition des phases
phases = data.groupby(["Y"]).groups
for i in phases.keys():
    print(str(i) + " : " + str(phases[i].size))
#0 : 4939
#1 : 1359
#2 : 16139
#3 : 13780
#4 : 7613    

    
def plot_vardensity(data):
    fig, ax = plt.subplots(figsize=(40,20))
    X = data.iloc[:, :-1]
    i = 1
    for k in X.columns:
        plt.subplot(9,7, i)
        sns.kdeplot(X[k], shade=True)
        #plt.title(k, loc='left', fontsize=18, fontweight=3)
        i = i +1 
        if (i==4*7+1):
            plt.figure()
            fig, ax = plt.subplots(figsize=(40,20))
        elif (i == 4*7 + 2*7 + 1):
            plt.figure()
            fig, ax = plt.subplots(figsize=(40,20))
    plt.figure()
    
plot_vardensity(data)

def make_awake_asleep(data):
    df = data.copy()
    df[df.Y !=0] = 1
    return df
    
data_simple = make_awake_asleep(data)

sns.countplot(x="Y", data=data_simple)    

# repartition des phases
phases = data_simple.groupby(["Y"]).groups
for i in phases.keys():
    print(str(i) + " : " + str(phases[i].size))

#0 : 4939
#1 : 38891    
    
####################


def generate_samples(df): 
        training, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])
        return [training, test]

training_simple, test_simple = generate_samples(data_simple)  #probleme !!!

sns.countplot(x="Y", data=training_simple)
sns.countplot(x="Y", data=test_simple)

sleep.iloc[:, 0].unique()

def sanity(simple):
    df_ = training_simple.iloc[:, -8:]
    df = df_.copy()
    awake = df.loc[df.Y==0]
    sleep = df.loc[df.Y !=0]    
    print("awake ", awake.head())
    print("sleep ", sleep.head())

sanity(training_simple)
sanity(test_simple)


def predwake(X, y, X_test, y_true):
    Lkappa_l1 = []
    Lkappa_l2 = []
    L = [0.00000001,0.0000001, 0.000001, 0.00001, 0.0001, 0.01, 1, 100]
    for C in L: 
        l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='liblinear')
        l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='liblinear')
        l1_LR.fit(X, y)
        l2_LR.fit(X, y)
        y_pred_l1 = l1_LR.predict(X_test)
        y_pred_l2 = l2_LR.predict(X_test)
        coef_l1_LR = l1_LR.coef_.ravel()
        coef_l2_LR = l2_LR.coef_.ravel()
        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
        kappa_l1 = cohen_kappa_score(y_true, y_pred_l1)
        kappa_l2 = cohen_kappa_score(y_true, y_pred_l2)
        Lkappa_l1.append(kappa_l1)
        Lkappa_l2.append(kappa_l2)
        print("C=%.8f" % C)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("L1 coef", coef_l1_LR)
        notScarseCoefs = []
        for i in range(0, len(coef_l1_LR)):
            if (coef_l1_LR[i] !=0):
                notScarseCoefs.append(X.columns[i])
        print("Not null coef for l1", notScarseCoefs)
        print("Precision L1: %.4f" % precision_score(y_true, y_pred_l1))
        print("recall L1: %.4f" % recall_score(y_true, y_pred_l1))
        print("kappa L1: %.4f" % kappa_l1)
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        print("Precision L2: %.4f" % precision_score(y_true, y_pred_l2))
        print("recall L2: %.4f" % recall_score(y_true, y_pred_l2))
        print("kappa L2: %.4f" % kappa_l2)
    fig, ax = plt.subplots()
    ax.set_title("kappa vs C for l1")
    ax.semilogx(L, Lkappa_l1)
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("kappa vs C for l2")
    ax.semilogx(L, Lkappa_l2)
    plt.show()
    return [Lkappa_l1, Lkappa_l2]

predwake(Xeeg1, ywake, Xeeg1_valid, ywake_true )



X14 = training_simple.iloc[:, :28]
X14_test = test_simple.iloc[:, :28]

predwake(X14, ywake, X14_test, ywake_true)

X63 = training_simple.iloc[:, :-1]
X63_test = test_simple.iloc[:, :-1]


predwake(X63, ywake, X63_test, ywake_true)
# C=0.01
# Sparsity with L1 penalty: 33.33%
# Precision L1: 0.8894
# recall L1: 0.9982
# kappa L1: 0.0370
# Sparsity with L2 penalty: 0.00%
# Precision L2: 0.8915
# recall L2: 0.9886
# kappa L2: 0.0658

#C=0.00000100
#kappa L1: 0.1877

# with C=0.00000010 and l1 not null coef are ['max_eeg_1', 'min_po_ir', 'min_po_r', 'max_po_r']

Xacc = training_simple.iloc[:, -22:-1]
Xacc_test = test_simple.iloc[:, -22:-1]
predwake(Xacc, ywake, Xacc_test, ywake_true)
#kappa L1: 0.1338 with c=1




L4 = ['max_eeg_1', 'min_po_ir', 'min_po_r', 'max_po_r']
X4 = pd.DataFrame()
X4_test = pd.DataFrame()
for i in L4:
    X4[i] = training_data[i]
    X4_test[i] = validation_data[i]

predwake(X4, ywake, X4_test, ywake_true)
#best kappa at about 0.14



#####################################
# awake should be seen on accelerometer_x

def plot_vardensity_awake(df_):
    df = df_.copy()
    awake = df.loc[df.Y==0]
    sleep = df.loc[df.Y !=0]
    for k in df.iloc[:, :-1].columns:
        fig, ax = plt.subplots(figsize=(30,15))
        plt.subplot(1,2, 1)
        plt.title(k + " awake")
        sns.kdeplot(awake[k], shade=True)
        plt.subplot(1,2, 2)
        sns.kdeplot(sleep[k], shade=True)
        plt.title(k + " sleep")
        plt.figure()

plot_vardensity_awake(data.iloc[:, -8:])

df_ = data.iloc[:, -8:]
df = df_.copy()
awake = df.loc[df.Y==0]
# awake row = 3951
sleep = df.loc[df.Y !=0]
sns.countplot(x="Y", data=df)

for i in sleep.columns:
    sns.countplot(x=i, data=sleep)
    plt.figure()



#Gradient tree boosting
####################

X_new = validation_data.iloc[:, :-1]
y_proba = log_reg.predict_proba(X_new)
y_pred = log_reg.predict(X_new)


confusion_matrix(ywake_true, y_pred)
precision_score(ywake_true, y_pred)
recall_score(ywake_true, y_pred)


#LogisticRegressionCV implements Logistic Regression with builtin cross-validation 

##########################
logit = sm.Logit(ywake, X)

  # fit the model
result = logit.fit()

#Recursive Feature Elimination


log_reg = LogisticRegression() 
log_reg.fit(X, ywake)


###################




log_reg = LogisticRegression() 
log_reg.fit(X, y)

coef = log_reg.coef_.ravel()
sparsity = np.mean(coef == 0) * 100

#LogisticRegression(C=C, penalty='l1', tol=0.01)

 

            
            
##########################
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