import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


N = 8 # number of subbands
L = 512 # length of filter
n = np.arange(L)
wc = (1.0/N)*np.pi # lowpass bandwidth

kw  = np.kaiser(L,8) # Kaiser window

filt = np.zeros((N, L))  # filters

# ideal low pass filter with rectangular window
filt[0] = np.sin(0.875*wc*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
# ideal high pass filter with rectangular window
filt[7] = highpass=np.sin(np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))-np.sin(0.875*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))

# Kaiser window passband = 0.04pi, ideal passband = 0.085pi
filt0 = np.sin(0.08*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))

# center is wc*(i+0.5)
for i in range(0,8):
   if 0<i & i<7:
      filt[i] = filt0*kw*np.cos(n*wc*(i+0.5))
   elif i==0:
      filt[i] = filt0*kw
   elif i==7:
      filt[i] = highpass*kw

f1,fs1 = sig.freqz(filt[0])    # frequency response of filters
f2,fs2 = sig.freqz(filt[1])
f3,fs3 = sig.freqz(filt[2])
f4,fs4 = sig.freqz(filt[3])
f5,fs5 = sig.freqz(filt[4])
f6,fs6 = sig.freqz(filt[5])
f7,fs7 = sig.freqz(filt[6])
f8,fs8 = sig.freqz(filt[7])

p1, = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
p2, = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
p3, = plt.plot(f3,20*np.log10(np.abs(fs3)+1e-6))
p4, = plt.plot(f4,20*np.log10(np.abs(fs4)+1e-6))
p5, = plt.plot(f5,20*np.log10(np.abs(fs5)+1e-6))
p6, = plt.plot(f6,20*np.log10(np.abs(fs6)+1e-6))
p7, = plt.plot(f7,20*np.log10(np.abs(fs7)+1e-6))
p8, = plt.plot(f8,20*np.log10(np.abs(fs8)+1e-6))

plt.title('Frequency response of filters')
plt.legend(handles = [p1,p2,p3,p4,p5,p6,p7,p8,],labels = ['filter1','filter2','filter3','filter4','filter5','filter6','filter7','filter8'])
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()


