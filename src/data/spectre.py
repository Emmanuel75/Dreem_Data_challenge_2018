from scipy import signal


from intervals import FloatInterval

FREQ_BANDS = collections.OrderedDict()
FREQ_BANDS['Delta'] = [0.0, 4.0]
FREQ_BANDS['Theta'] = [4.0, 8.0]
FREQ_BANDS['Alpha'] = [8.0, 16.0]
FREQ_BANDS['Beta'] = [16.0, 32.0]
FREQ_BANDS['Gamma'] = [32.0, 100.0]
FREQ_BANDS['Above100Hz'] = [100.0, 9e3]

sig = eeg_1.iloc[55] #125Hz signal

plt.plot(sig)

#f, t, Sxx = signal.spectrogram(trace.data, fs=40, nperseg=200, window=('hamming'), noverlap=100)
#fs = frequency donc pour eeg: 125. 
fs = 125

timeIntervalSec=2 # fenetre sur deux secondes 

#npserg : length of each segment 
# Length of segment for each fft = nb of timepoints = timeIntervalSec * Sampling frequency
nperseg = int(fs * timeIntervalSec)


#channelName


#Length of segment for each fft = nb of timepoints = timeIntervalSec * Sampling frequency



#y = df_all[start:end][channelName].values

freqs, midtimes, spectroArray = signal.spectrogram(sig, fs, nperseg=nperseg, noverlap=0)
#The spectrogram is plotted as a colormap (using imshow)
plot.imshow(spectroArray)

fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis')
#vmin = 20*np.log10(np.max(x)) - 40  # hide anything below -40 dBc
#cmap.set_under(color='k', alpha=None)
Pxx, freqs, bins, im = ax.specgram(sig, Fs=fs, NFFT=nperseg, noverlap=0 )
plot.xlabel('Time')
plot.ylabel('Frequency')
fig.colorbar(im)

#freqs, times, spectrogram = signal.spectrogram(sig, fs=1)

#freqs, psd = signal.welch(sig, fs=1)

plt.semilogx(freqs, psd)

#################################################*
sig = eeg_1.iloc[55] #125Hz signal
samples = eeg_1.iloc[55]
sample_rate= 125
timeIntervalSec=2
fs = 125
nperseg = int(fs * timeIntervalSec)




def log_specgram(audio, sample_rate=125, window_size=250,
                 step_size=10, eps=1e-10):
    nperseg = int(sample_rate * timeIntervalSec)
    noverlap = 0
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


freqs, times, spectrogram = log_specgram(sig, sample_rate=125, window_size=250)
toto = pd.DataFrame(spectrogram)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave')
ax1.set_ylabel('Amplitude')
ax1.xaxis.set_ticks(np.linspace(0, len(samples), num=30))
ax1.xaxis.set_ticklabels(range(0,30))
ax1.plot(samples)
#ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

ax2 = fig.add_subplot(212)
im= ax2.imshow(spectrogram.T, aspect='auto', origin='lower')
#plt.matshow(spectrogram.T, aspect='auto', origin='lower')
#ax2.set_yticks(freqs[::16])
#ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ')
ax2.set_ylabel('Freqs in 1/2 Hz')
ax2.set_xlabel('Seconds')
ax2.xaxis.set_ticks(np.linspace(0, len(times), num=len(times)))
ax2.xaxis.set_ticklabels(times)
#fig.colorbar(im)
fig.show()


fig = plt.figure(figsize=(14, 8))
ax2 = fig.add_subplot(212)
im= ax2.imshow(spectrogram.T, aspect='auto', origin='lower')
#plt.matshow(spectrogram.T, aspect='auto', origin='lower')
#ax2.set_yticks(freqs[::16])
#ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ')
ax2.set_ylabel('Freqs in 1/2 Hz')
ax2.set_xlabel('Seconds')
ax2.xaxis.set_ticks(np.linspace(0, len(times), num=len(times)))
ax2.xaxis.set_ticklabels(times)
#ax2.yaxis.set_ticklabels(freqs[::20])
fig.colorbar(im)
fig.show()

#################################################

sig = eeg_1.iloc[55] #125Hz signal
samples = eeg_1.iloc[55]
sample_rate= 125
timeIntervalSec=2
#fs = 125
#nperseg = int(fs * timeIntervalSec)




def my_spectrogram(sig, sample_rate=125, timeIntervalSec=2):
    nperseg = int(sample_rate * timeIntervalSec)
    noverlap = 0
    freqs, times, spec = signal.spectrogram(sig,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=sample_rate*timeIntervalSec,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times,  spec.astype(np.float32)

    
freqs, times, spectrogram = my_spectrogram(sig)
toto = pd.DataFrame(spectrogram)

FREQ_BANDS = collections.OrderedDict()
FREQ_BANDS['Delta'] = [0.0, 4.0]
FREQ_BANDS['Theta'] = [4.0, 8.0]
FREQ_BANDS['Alpha'] = [8.0, 16.0]
FREQ_BANDS['Beta'] = [16.0, 32.0]
FREQ_BANDS['Gamma'] = [32.0, 100.0]
FREQ_BANDS['Above100Hz'] = [100.0, 9e3]

def spectrogram_by_eeg_bandwidth(sig, sample_rate=125, timeIntervalSec=2):
    delta = FloatInterval.from_string('[0, 4.0)')
    theta = FloatInterval.from_string('[4.0, 8.0)')
    alpha = FloatInterval.from_string('[8.0, 16.0)')
    beta = FloatInterval.from_string('[16.0, 32.0)')
    gamma = FloatInterval.from_string('[32.0, 100.0)')
    above100Hz = FloatInterval.from_string('[100.0,)')
    #
    nperseg = int(sample_rate * timeIntervalSec)
    noverlap = 0
    freqs, times, spec = signal.spectrogram(sig,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=sample_rate*timeIntervalSec,
                                    noverlap=noverlap,
                                    detrend=False)
    #Edelta, Etheta, Ealpha, Ebeta, Egamma, Eabove100Hz = np.zeros(6)
    result = pd.DataFrame(np.zeros(spec.shape[1]*6).reshape((6, spec.shape[1])), index=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Above100Hz'])
    #
    for i in range(0, spec.shape[0]):
        for j in range(0, spec.shape[1]):
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
            elif freqs[i] in above100Hz:
                print(str((i, j)) + "added tp above100Hz")
                result.loc["Above100Hz", j]= result.loc["Above100Hz", j] + spec[i, j]
            else:
                print("error at cell " + str((i, j)))
    return result


 spec = np.log(spectrogram_by_eeg_bandwidth(sig).iloc[:-1, :])

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave')
ax1.set_ylabel('Amplitude')
ax1.xaxis.set_ticks(np.linspace(0, len(samples), num=30))
ax1.xaxis.set_ticklabels(range(0,30))
ax1.plot(sig)
#ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

ax2 = fig.add_subplot(212)
im= ax2.imshow(spec, aspect='auto', origin='lower')
#plt.matshow(spectrogram.T, aspect='auto', origin='lower')
#ax2.set_yticks(freqs[::16])
#ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ')
#ax2.set_ylabel('Freqs in 1/2 Hz')
ax2.set_xlabel('Seconds')
ax2.xaxis.set_ticks()
#ax2.xaxis.set_ticklabels(times)
ax2.yaxis.set_ticklabels(['0', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
#fig.colorbar(im)


ax3 = fig.add_subplot()
ax3.plot(spec.T["Alpha"], label = "Alpha")
ax3.legend()

fig.show()




###

plt.semilogy(titi.T)

fig = plt.figure(figsize=(14, 8))
ax2 = fig.add_subplot(212)
im= plt.semilogy(titi.T)
#plt.matshow(spectrogram.T, aspect='auto', origin='lower')
#ax2.set_yticks(freqs[::16])
#ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ')
ax2.set_ylabel('Energy')
ax2.set_xlabel('Seconds')
plt.legend()
#ax2.yaxis.set_ticklabels(freqs[::20])
#fig.colorbar(im)
fig.show()

sns.lineplot(titi.T)


plt.plot(rx, Lk, label = "kappa")
plt.plot(rx, La, label = "accuracy")
plt.legend(loc='lower right')



fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(np.log(titi.iloc[:-1, :]), aspect='auto', origin='lower')
ax.yaxis.set_ticklabels(['0', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
fig.colorbar(cax)
#im = plt.matshow(np.log(titi.iloc[:-1, :]), aspect='auto', origin='lower')
#######################

def generate_columns_names(L=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Above100Hz'], n=15):
    r = "["
    for i in L:
        for j in range(0,n):
            if (i == L[-1]) and (j==n-1):
                r = r + "'" + str(i)+ str(j) + "']"
            else:
                r = r + "'" + str(i)+ str(j) + "',"
    return eval(r)


def make_eeg_spectogram_dataframe(eeg, timeIntervalSec):
    columns = generate_columns_names(n = 30//timeIntervalSec)
    df = pd.DataFrame(columns = columns)
    for i in range(0, eeg.shape[0]):
        spec = spectrogram_by_eeg_bandwidth(eeg.iloc[i,:], 
            sample_rate=125, timeIntervalSec=timeIntervalSec)
        t = spec.values.reshape(1, spec.shape[0]*spec.shape[1])
        #df.loc[i] = [j for j in t[0]]
        df.loc[i] = t[0]
    return df

        
