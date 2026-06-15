import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Define simulation parameters
N = 12

# Create two discrete sequences (length <= N)
x1_seq = np.array([1.0, 1.0, 1.0, 1.0])
x2_seq = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# Compute the time-domain circular convolution
y_out = circonvt(x1_seq, x2_seq, N)

# Construct uniform time axis arrays for display purposes
n_axis = np.arange(0, N)
x1_padded = np.pad(x1_seq, (0, N - len(x1_seq)), 'constant')
x2_padded = np.pad(x2_seq, (0, N - len(x2_seq)), 'constant')

# Plotting the signals
plt.figure(figsize=(12, 8))

# Subplot 1: Input Sequence x1[n]
plt.subplot(3, 1, 1)
plt.stem(n_axis, x1_padded, basefmt=" ")
plt.plot(n_axis, x1_padded, color='blue', alpha=0.4, linestyle='-')
plt.title('Padded Input Sequence x1[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Input Sequence x2[n]
plt.subplot(3, 1, 2)
plt.stem(n_axis, x2_padded, linefmt='C1-', markerfmt='C1o', basefmt=" ")
plt.plot(n_axis, x2_padded, color='orange', alpha=0.4, linestyle='-')
plt.title('Padded Input Sequence x2[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Circular Convolution Result y[n]
plt.subplot(3, 1, 3)
plt.stem(n_axis, y_out, linefmt='purple', markerfmt='mo', basefmt=" ")
plt.plot(n_axis, y_out, color='purple', alpha=0.4, linestyle='-')
plt.title(f'{N}-Point Circular Convolution Output y[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()