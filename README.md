In this exercise we have designed an audio filterbank using windowing and modulation. 

# In filter "des.py" we have
* Created an ideal low pass filter using rectangular window
* Convolved the Ideal low pass filter with Impulse centered on the band pass center frequency
* Plotted the frequency respose of the filter bank

# In "freq. res. diff. win-types.py" we have 
* Created 5 different windows
* Created an ideal low pass filter
* Convolved the ideal low pass with the windows one by one
* Plotted the resultant filters

# In "ori. Vs Reco. Signal.py" we have
* Applied the filterbank on the audio file
* Plotted the original and reconstructed signals

# Usage
* python des.py
* python freq. res. diff. win-types.py
* python ori. Vs Reco. Signal.py