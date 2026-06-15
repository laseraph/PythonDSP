import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import lfilter

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Define an arbitrary 4th-order Direct-Form IIR filter system
b_direct = [1.0, 0.4, 0.3, 0.2, 0.1]
a_direct = [1.0, -0.5, 0.4, -0.3, 0.2]

# 1. Convert Direct-Form parameters to Cascade Second-Order Sections
b0, B_mat, A_mat = dir2cas(b_direct, a_direct)

# Define an impulse input sequence to capture the filter's profile over time
n_samples = 30
time_axis = np.arange(0, n_samples)
x_impulse = np.where(time_axis == 0, 1.0, 0.0)

# 2. Compute output using Cascade filtering architecture
y_cascade = casfiltr(b0, B_mat, A_mat, x_impulse)

# 3. Compute baseline standard Direct-Form filtering for visual comparison
y_direct = lfilter(b_direct, a_direct, x_impulse)

# Plotting the outputs
plt.figure(figsize=(12, 6))

# Subplot 1: Cascade Realization Output
plt.subplot(2, 1, 1)
plt.stem(time_axis, y_cascade, linefmt='C0-', markerfmt='C0o', basefmt=" ")
plt.plot(time_axis, y_cascade, color='blue', alpha=0.4, linestyle='-')
plt.title('Filter Impulse Response via Cascade Realization (casfiltr)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Direct-Form Baseline Output
plt.subplot(2, 1, 2)
plt.stem(time_axis, y_direct, linefmt='purple', markerfmt='mo', basefmt=" ")
plt.plot(time_axis, y_direct, color='purple', alpha=0.4, linestyle='-')
plt.title('Filter Impulse Response via Direct-Form Baseline (lfilter)')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()