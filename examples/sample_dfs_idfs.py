import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# =============================================================================
# 1. SAMPLE CODE: Discrete Fourier Series (DFS)
# =============================================================================

# Define fundamental period
N = 20

# Create one period of a periodic square wave signal (5 ones followed by 15 zeros)
xn = np.zeros(N)
xn[0:5] = 1.0

# Computing the DFS coefficients
Xk = dfs(xn, N)

# Calculate the magnitude and phase of the DFS coefficients
dfs_mag = np.abs(Xk)
dfs_phase = np.angle(Xk)

# Plotting the DFS Results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.stem(np.arange(N), xn, basefmt=" ", label='Discrete Samples')
plt.plot(np.arange(N), xn, color='blue', alpha=0.5, linestyle='-', linewidth=1.5, label='Connecting Curve')
plt.title('One Period of Periodic Signal x[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(3, 1, 2)
plt.stem(np.arange(N), dfs_mag, linefmt='C1-', markerfmt='C1o', basefmt=" ")
plt.plot(np.arange(N), dfs_mag, color='orange', alpha=0.6, linestyle='-', linewidth=1.5)
plt.title('Magnitude of DFS Coefficients |Xk|')
plt.xlabel('k')
plt.ylabel('Magnitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(3, 1, 3)
plt.stem(np.arange(N), dfs_phase, linefmt='red', markerfmt='ro', basefmt=" ")
plt.plot(np.arange(N), dfs_phase, color='red', alpha=0.4, linestyle='-', linewidth=1.5)
plt.title('Phase of DFS Coefficients')
plt.xlabel('k')
plt.ylabel('Radians')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.suptitle('Discrete Fourier Series (DFS) Analysis with Envelopes', y=1.02, fontsize=14)
plt.show()


# =============================================================================
# 2. SAMPLE CODE: Inverse Discrete Fourier Series (IDFS)
# =============================================================================

# Computing the reconstruction from the DFS coefficients using IDFS
xn_reconstructed = idfs(Xk, N)

# Take the real part of the reconstructed signal (to remove negligible numerical noise)
xn_reconstructed_real = np.real(xn_reconstructed)

# Plotting the IDFS Verification Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.stem(np.arange(N), xn, basefmt=" ")
plt.plot(np.arange(N), xn, color='blue', alpha=0.5)
plt.title('Original Signal x[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.stem(np.arange(N), xn_reconstructed_real, linefmt='purple', markerfmt='mo', basefmt=" ")
plt.plot(np.arange(N), xn_reconstructed_real, color='purple', alpha=0.5)
plt.title('Reconstructed Signal from IDFS')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.suptitle('Inverse DFS Verification with Envelopes', y=1.05, fontsize=14)
plt.show()