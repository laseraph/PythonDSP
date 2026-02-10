import numpy as np
import matplotlib.pyplot as plt

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.sigfunctions import *

# Plotting $x[n]=u(n)-u(n-10)$
n1 = np.arange(0,20)
x1 = 2*(0.8**n1)*(stepseq(0,19,0)[1]-stepseq(0,19,20)[1])

plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.stem(n1,x1,markerfmt='ro',linefmt='b-')
plt.title(f'Input Sequence $x[n]=u(n)-u(n-10)$')
plt.xlabel(f'$n$')
plt.ylabel(f'$y[n]$')
plt.grid(True, linestyle='--', alpha=0.5)

# Plotting $h[n]=0.9^nu(n)$
n2 = np.arange(0,51)
h1 = (0.9**n2)*stepseq(0,50,0)[1]

plt.subplot(3,1,2)
plt.stem(n2,h1,markerfmt='ro',linefmt='b-')
plt.title(f'Impulse Sequence $h[n]=0.9^nu(n)$')
plt.xlabel(f'$n$')
plt.ylabel(f'$h[n]$')
plt.grid(True, linestyle='--', alpha=0.5)

y, ny = conv_ext(x1,n1,h1,n2)

plt.subplot(3,1,3)
plt.stem(ny,y,markerfmt='ro',linefmt='b-')
plt.title(f'Convolved Sequence $y[n]=x[n]*h[n]$')
plt.xlabel(f'$n$')
plt.ylabel(f'$y[n]=x[n]*h[n]$')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()