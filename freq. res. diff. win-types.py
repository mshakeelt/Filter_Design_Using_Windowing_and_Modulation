import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


N = 8 # number of subbands
L = 128 # length of filter
n = np.arange(L)
wc = (1.0/N)*np.pi # lowpass bandwidth


w1 = np.ones(L) # rectangular window
w2 = 0.5 - 0.5*np.cos(2*np.pi/L*(n+0.5)) # Hanning window
w3 = np.sin(np.pi/L*(n+0.5))  # sine window
w4 = np.kaiser(L,8) # Kaiser window
w5 = np.sin(np.pi/2*np.sin(np.pi/L*(n+0.5))**2)  # kaiser besel derived

filt = np.zeros((N, L))  # filters

# ideal low pass filter with rectangular window
filt[0] = np.sin(wc*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
window = np.random.rand(L)

ideal_low = np.sin(wc*(n-(L-1)/2))/(wc*(n-(L-1)/2))
L_P = ideal_low * w1 

f,fs = sig.freqz(L_P)
print(20*np.log10(np.abs(fs)+1e-6))
print(len(fs))
plt.plot(f,20*np.log10(np.abs(fs)+1e-6))
plt.show()

plt.plot(L_P)

plt.show()

h1 = filt[0]*w1
plt.plot(h1)
plt.xlim(-10,150)

h2 = filt[0]*w2
plt.plot(h2)
plt.xlim(-10,150)


h3 = filt[0]*w3
plt.plot(h3)
plt.xlim(-10,150)

h4 = filt[0]*w4
plt.plot(h4)
plt.xlim(-10,150)

h5 = filt[0]*w5
plt.plot(h5)
plt.xlim(-10,150)

plt.show()


f1,fs1 = sig.freqz(h1)
f2,fs2 = sig.freqz(h2)
f3,fs3 = sig.freqz(h3)
f4,fs4 = sig.freqz(h4)
f5,fs5 = sig.freqz(h5)

p1, = plt.plot(f1,20*np.log10(np.abs(fs1)+1e-6))
p2, = plt.plot(f2,20*np.log10(np.abs(fs2)+1e-6))
p3, = plt.plot(f3,20*np.log10(np.abs(fs3)+1e-6))
p4, = plt.plot(f4,20*np.log10(np.abs(fs4)+1e-6))
p5, = plt.plot(f5,20*np.log10(np.abs(fs5)+1e-6))

plt.title('Frequency Response of filters of Different Window Types')
plt.legend(handles = [p1,p2,p3,p4,p5,],labels = ['Rectangular','Hanning','Sine','Kaiser','KBD'])
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()