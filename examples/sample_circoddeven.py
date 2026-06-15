import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Define an asymmetrical finite test sequence x[n] = n + 1 for N = 10
N = 10
n_vals = np.arange(0, N)
x = n_vals + 1.0

# Decompose the sequence using circevod
xec, xoc = circevod(x)

# Plotting the decomposition
plt.figure(figsize=(12, 8))

# Subplot 1: Original Signal
plt.subplot(3, 1, 1)
plt.stem(n_vals, x, basefmt=" ")
plt.plot(n_vals, x, color='blue', alpha=0.4, linestyle='-')
plt.title('Original Sequence x[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Circular-Even Component
plt.subplot(3, 1, 2)
plt.stem(n_vals, xec, linefmt='C1-', markerfmt='C1o', basefmt=" ")
plt.plot(n_vals, xec, color='orange', alpha=0.4, linestyle='-')
plt.title('Circular-Even Component xec[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Circular-Odd Component
plt.subplot(3, 1, 3)
plt.stem(n_vals, xoc, linefmt='red', markerfmt='ro', basefmt=" ")
plt.plot(n_vals, xoc, color='red', alpha=0.4, linestyle='-')
plt.title('Circular-Odd Component xoc[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()