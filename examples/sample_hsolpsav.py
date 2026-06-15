import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Define inputs and block size
N_block = 8
x_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
h_filter = np.array([1.0, -1.0, 1.0])

# 1. Compute convolution using the Overlap-Save method
y_overlap_save = hsolpsav(x_signal, h_filter, N_block)

# 2. Compute baseline standard linear convolution for error verification
# (The total length of standard linear convolution is len(x) + len(h) - 1)
y_linear, n_linear = conv_ext(x_signal, np.arange(len(x_signal)), h_filter, np.arange(len(h_filter)))

# Dynamic time axis vector matching the actual length of the overlap-save output array
n_axis = np.arange(len(y_overlap_save))

# Plotting the comparison
plt.figure(figsize=(12, 6))

# Subplot 1: High-Speed Overlap-Save Result
plt.subplot(2, 1, 1)
plt.stem(n_axis, y_overlap_save, linefmt='C0-', markerfmt='C0o', basefmt=" ")
plt.plot(n_axis, y_overlap_save, color='blue', alpha=0.4, linestyle='-')
plt.title('Convolution via High-Speed Overlap-Save method (hsolpsav)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Standard Linear Convolution Benchmark
plt.subplot(2, 1, 2)
plt.stem(n_linear, y_linear, linefmt='purple', markerfmt='mo', basefmt=" ")
plt.plot(n_linear, y_linear, color='purple', alpha=0.4, linestyle='-')
plt.title('Standard Linear Convolution Baseline (conv_ext)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()