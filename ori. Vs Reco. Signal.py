import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sound import *

N = 8 # number of subbands
L = 512 # length of filter
n = np.arange(L)
wc = (1.0/N)*np.pi # lowpass bandwidth

# reading audio
fs, s = wav.read("Track32.wav")
s1 = s[:,0]   #channel 1


kw = np.sin(np.pi/2*np.sin(np.pi/L*(n+0.5))**2) # KBD window

filt = np.zeros((N, L))  # filters

# ideal low pass filter with rectangular window
filt[0] = np.sin(wc*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
# ideal high pass filter with rectangular window
filt[7] =np.sin(np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))-np.sin(0.875*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))

# Kaiser window passband = 0.04pi, ideal passband = 0.085pi
filt0 = np.sin(0.04*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))

# center is wc*(i+0.5)
for i in range(1,7):
   filt[i] = filt0*kw*np.cos(n*wc*(i+0.5))



# Analysis filter bank
filtered1 = np.zeros((N,len(s1)))
for i in range(N):
   filtered1[i] = sig.lfilter(filt[i],1,s1)

# downsampling

ds = np.zeros((N,int(len(s1)/N)))
for i in range(N):
   ds[i] = filtered1[i,0::N]

#Synthesis filter bank
# upsampling
us = np.zeros((N,len(s1)))
for i in range(N):
   us[i,0::N] = ds[i]

# filtering
filtered2 = np.zeros((N,len(s1)))
for i in range(N):
   filtered2[i] = sig.lfilter(filt[i],1,us[i])

# reconstructed signal
sys = filtered2[0]+filtered2[1]+filtered2[2]+filtered2[3]+filtered2[4]+filtered2[5]+filtered2[6]+filtered2[7]
#sound(sys, fs)

s1 = s1/max(s1)
sys = sys/max(sys)
l1, = plt.plot(s1, color='red')
l2, = plt.plot(sys, color='blue')
plt.legend(handles = [l1,l2,], labels = ['Original','Reconstructed'])
plt.title('Original signal and reconstructed signal')
plt.show()
