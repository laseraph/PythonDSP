import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Define parameters
N = 10                  # Circular buffer size
m = 3                   # Shift right by 3 samples

# Create a sample input sequence of length <= N (e.g., length 7)
x_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])

# Perform the circular time shift
y_shifted = cirshftt(x_input, m, N)

# Generate a time index for plotting the full buffer length N
n_range = np.arange(0, N)

# Pad the original signal manually just for an accurate visual baseline comparison
x_padded = np.pad(x_input, (0, N - len(x_input)), 'constant')

# Plotting the signals
plt.figure(figsize=(12, 5))

# Subplot 1: Original Padded Sequence
plt.subplot(1, 2, 1)
plt.stem(n_range, x_padded, basefmt=" ")
plt.plot(n_range, x_padded, color='blue', alpha=0.4, linestyle='-')
plt.title('Original Padded Sequence x[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Circularly Shifted Sequence
plt.subplot(1, 2, 2)
plt.stem(n_range, y_shifted, linefmt='purple', markerfmt='mo', basefmt=" ")
plt.plot(n_range, y_shifted, color='purple', alpha=0.4, linestyle='-')
plt.title(f'Circularly Shifted Sequence y[n] (m = {m})')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()