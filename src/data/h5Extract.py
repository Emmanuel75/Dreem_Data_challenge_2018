import h5py
import pandas as pd


######################




#Datasets description: 43830 train samples for 20592 test samples


#sleep classification follows the AASM recommendations and was labeled by a single expert.
# 0 : Wake
# 1 : N1 (light sleep)
# 2 : N2
# 3 : N3 (deep sleep)
# 4 : REM (paradoxal sleep)

#4 EEG channels sampled at 125Hz (4 x 125 x 30 = 15000 size per sample)
# for each eeg, [43830 rows x 3750 columns]: each sample is 30 secs with 125 Hz
# 2 Pulse Oxymeter channels (red and infra-red) sampled at 50 Hz (2 x 50 x 30 = 3000 size per sample)
# 3 Accelerometer channels sampled at 50Hz (3 x 50 x 30 = 4500 size per sample)


############
#extract data from h5 and creates features

def build_features(h5filename, output=[] , dataPath="C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\raw\\"):
    h5file= dataPath + h5filename
    h5 = h5py.File(h5file, "r")
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
    h5keys = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'po_ir', 'po_r', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z']
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
    if (output!=[]):
        trainOutput = pd.read_csv(dataPath + output, sep=";")
        data["Y"] = trainOutput["label"]
    return data

# write back to data\iterim
data_train = build_features("train.h5", "challenge_fichier_de_sortie_dentrainement_classification_en_stade_de_sommeil_a_laide_de_signaux_mesures_par_le_bandeau_dreem.csv")   
data_test = build_features("test.h5")
data_train.to_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\featuresTrain.xlsx')
data_test.to_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\featuresTest.xlsx')



##############################
# transform target in a 0/1 target (awake/sleep)
def as_binomial (data_train):
    df = data_train.copy()
    df.loc[df.Y !=0, "Y"] = 1
    return df
    
    
with_binomial_target = as_binomial(data_train)
with_binomial_target.to_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\binomial_target.xlsx')

df = pd.read_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\binomial_target.xlsx')