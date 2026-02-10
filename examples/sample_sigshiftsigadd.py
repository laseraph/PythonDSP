import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import * 

# Define the time vector n and discrete signal x[n]
n1 = np.arange(-2,10+1)
x = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])

# Try to solve x(n)=2x(n-5)-3x(n+4)
k1 = 5
yt1,nt1 = sigshift(x,n1,k1)

k2 = -4
yt2,nt2 = sigshift(x,n1,k2)

# Call the function sigadd:
y_add, n_add = sigadd(2*yt1,nt1,-3*yt2,nt2)

# Plot the discrete signal x[n]
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.stem(nt1,yt1)
plt.title('First Term')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n-5]$')

# Plot the even parts of the signal x[even]
plt.subplot(3,1,2)
plt.stem(nt2,yt2)
plt.title('Second Term')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n+4]$')

# Plot the odd parts of the signal x[odd]
plt.subplot(3,1,3)
plt.stem(n_add,y_add)
plt.title(f'Function $x(n)=2x(n-5)-3x(n+4)$')
plt.xlabel(f'$n$')
plt.ylabel(f'$x[n]$')

plt.tight_layout()
plt.show()