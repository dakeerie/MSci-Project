import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
#Time domain signal
Fs = 2000 #kHz sampling frequency
tstep = 1/Fs #sample time interval 
f0 = 100 #Hz frequency idealised sine wave

N = int(10*Fs/f0) #number of samples of the signals
t = np.linspace(0, (N - 1)*tstep, N)
fstep = Fs/N #sampling frequency divided by number of samples gives frequency interval

f = np.linspace(0, (N - 1)*fstep, N) #frequency domain x axis

y = np.sin(2*np.pi*t*f0) + np.cos(2*np.pi*3*t*f0)#signal

X = np.fft.fft(y) #complex valued
#For now only interested in the magnitude of the output
X_mag = np.abs(X)/len(y) #normalise by number of samples? Average? 
X_phase = np.angle(X)
f_plot = f[0: int(N/2 + 1)]
X_mag_plot = 2*X_mag[0: int(N/2 + 1)]
X_mag_plot[0] = X_mag_plot[0]/2 #DC component does not need to be multiplied by 2?
X_phase_plot = X_phase[0:int(N/2 + 1)]

fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
ax1.plot(t, y, '.-')
ax2.plot(f_plot, X_mag_plot, '.-')
ax3.plot(f_plot, X_phase_plot, '.-')
plt.show()