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

# Call the function evenodd:
xe, xo, m = evenodd(x, n)

# Plot the discrete signal x[n]
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.stem(n,x)
plt.title('Discrete signal x[n]')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n]$')

# Plot the even parts of the signal x[even]
plt.subplot(3,1,2)
plt.stem(m,xe)
plt.title('Even components of the discrete signal x[n]')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[even]$')

# Plot the odd parts of the signal x[odd]
plt.subplot(3,1,3)
plt.stem(m,xo)
plt.title('Odd components of the discrete signal x[n]')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[odd]$')

plt.tight_layout()
plt.show()