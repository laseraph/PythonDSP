import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils.sigfunctions import * 

# Define the time vector n and discrete signal x[n]
n = np.arange(0,20+1)
x = (0.9**n)*stepseq(0,20,0)[1]
N = len(n) # n-point DFT, it is recommended to use powers of 2

# Call the function evenodd:
Xk = DFT(x,N)
mag = np.abs(Xk)
phase = np.angle(Xk)

# Plot the discrete signal x[n]
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.stem(n,x)
plt.title(r'Signal $x[n] = 0.9^n u[n]$')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n]$')
plt.grid(True, linestyle='--', alpha=0.5)

k = np.arange(N)
plt.subplot(1,3,2)
plt.stem(k, mag) # you have the choice of using plt.plot(k, mag) but it is not recommended due to interpolation of the values
plt.title('Magnitude of DFT')
plt.xlabel('Frequency')
plt.ylabel('|X|')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1,3,3)
plt.stem(k, phase) # you have the choice of using plt.plot(k, mag) but it is not recommended due to interpolation of the values
plt.title('Phase of DFT')
plt.xlabel('Phase')
plt.ylabel('radians/pi')
plt.grid(True, linestyle='--', alpha=0.5)

x_restored = IDFT(Xk, N)
x_restored_real = np.real(x_restored)
mag_n = np.abs(x_restored_real)
phase_n = np.imag(x_restored_real)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.stem(n,x_restored)
plt.title(r'Reconstructed x[n] via IDFT')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n]$')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.stem(k, phase_n) # you have the choice of using plt.plot(k, mag) but it is not recommended due to interpolation of the values
plt.title('Phase of IDFT')
plt.xlabel('Phase')
plt.ylabel('radians/pi')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()