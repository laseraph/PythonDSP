import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Path configuration ---
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils.sigfunctions import * 

# Define the time vector n and discrete signal x[n]
n = np.arange(0, 21) # 0 to 20 inclusive

x = (0.9**n) * stepseq(0, 20, 0)[1] 
N = 32 # For Radix-2 FFT, N should be a power of 2. For standard DFT, N=21 is fine. You can override this.

# --- Forward Transform (DFT) ---
Xk = DFT(x, N)
mag = np.abs(Xk)
phase = np.angle(Xk)

# --- Plot Original Signal and DFT ---
plt.figure(figsize=(15, 5))

# Plot 1: Original Time-Domain Signal
plt.subplot(1, 3, 1)
plt.stem(n, x)
plt.title(r'Signal $x[n] = 0.9^n u[n]$')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True, linestyle='--', alpha=0.5)

# Plot 2: DFT Magnitude
k = np.arange(N)
plt.subplot(1, 3, 2)
plt.stem(k, mag) 
plt.title('Magnitude of DFT')
plt.xlabel('Frequency Bin (k)')
plt.ylabel('|X(k)|')
plt.grid(True, linestyle='--', alpha=0.5)

# Plot 3: DFT Phase
plt.subplot(1, 3, 3)
plt.stem(k, phase) 
plt.title('Phase of DFT')
plt.xlabel('Frequency Bin (k)')
plt.ylabel('Phase (radians)')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# --- Inverse Transform (IDFT) ---
x_restored = IDFT(Xk, N)
# Discard minor imaginary rounding errors from the computation
x_restored_real = np.real(x_restored) 

# --- Inverse Transform (IDFT) ---
x_restored = IDFT(Xk, N)
# Discard minor imaginary rounding errors from the computation
x_restored_real = np.real(x_restored) 

# Create a new time vector that matches the padded length N
n_restored = np.arange(N) 

# --- Plot Reconstructed Signal ---
plt.figure(figsize=(6, 5))

# Use n_restored instead of n!
plt.stem(n_restored, x_restored_real)
plt.title(f'Reconstructed x[n] via IDFT (N={N})')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()