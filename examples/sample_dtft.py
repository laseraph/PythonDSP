import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Plotting $h[n]=0.9^nu(n)$
n1 = 0
n2 = 19
n_vals = np.arange(n1,n2+1)

# Function is x[n] = 2*0.8^n[u(n)-u(n-20)]
x = 2*(0.8**n_vals)*(stepseq(0,19,0)[1]-stepseq(0,19,20)[1])

# Frequency vector k and step size
k_array = np.arange(0,501)
delta_val = np.pi/500

# Computing the DTFT of the function
w_array, X_omega = DTFT(x,n1,n2,k_array,delta_val)

# Calculate the real part, imaginary part, magnitude and phase of the DTFT signal
rpart = np.real(X_omega)
ipart = np.imag(X_omega)
mag = np.abs(X_omega)
phase = np.angle(X_omega)

# Plotting the signal
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.plot(w_array/np.pi, rpart, linewidth=2)
plt.title('Real Part of DTFT')
plt.xlabel('n')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2,2,2)
plt.plot(w_array/np.pi, ipart, linewidth=2, color='red')
plt.title('Imaginary Part of DTFT')
plt.xlabel('n')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2,2,3)
plt.plot(w_array/np.pi, mag, linewidth=2)
plt.title('Magnitude of DTFT')
plt.xlabel('Frequency')
plt.ylabel('|X|')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2,2,4)
plt.plot(w_array/np.pi, phase, linewidth=2, color='red')
plt.title('Phase of DTFT')
plt.xlabel('Phase')
plt.ylabel('radians/pi')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()