# **PythonDSP**

**Python-based Digital Signal Processing Materials for use in DSP classes**

This repository contains a collection of Python functions designed to simulate fundamental Digital Signal Processing (DSP) operations. It is intended for students and educators to visualize and manipulate discrete-time signals, replicating standard MATLAB DSP toolboxes in a Pythonic environment.

## **📦 Dependencies**

To run the scripts and functions in this repository, you will need:

* **NumPy**: For array manipulation and mathematical operations.  
* **Matplotlib**: For plotting and visualizing signals.
* **SciPy**: For advanced signal processing operations and filter functions.

To install the required libraries, run:  
```python
pip install numpy matplotlib scipy
```
## **🚀 Usage**

Import the functions from the main module sigfunctions.py into your script:  
```python
import numpy as np  
import matplotlib.pyplot as plt  
from sigfunctions import stepseq, sigshift, sigadd
```
## **📚 Function Reference**

### **Signal Generation**

#### **impseq(n1, n2, n0)**

Generates a unit sample sequence (impulse) $\delta[n-n_0]$ over the interval $n_1 \leq n \leq n_2$.

* **Returns:** n (time vector), x (signal array)

#### **stepseq(n1, n2, n0)**

Generates a unit step sequence $u[n-n_0]$ over the interval $n_1 \leq n \leq n_2$.

* **Returns:** n (time vector), x (signal array)

### **Signal Operations**

#### **sigshift(x, m, k)**

Shifts a signal $x[n]$ by $k$ units, resulting in $y[n] = x[n-k]$.

* **Returns:** y (shifted signal), n (new time vector)

#### **sigfold(x, n)**

Folds (time-reverses) a signal about $n=0$, resulting in $y[n] = x[-n]$.

* **Returns:** n (folded time vector), y (folded signal)

#### **evenodd(x, n)**

Decomposes a real signal $x[n]$ into its even and odd components.

* **Returns:** xe (even part), xo (odd part), m (symmetric time vector)

### Circular Operations

#### **circevod(x)**

Decomposes a real discrete sequence into its circular-even and circular-odd components.

* **Returns**: xec (circular-even part), xoc (circular-odd part)

#### **cirshftt(x, m, N)**

Performs a time-domain circular shift of an input sequence $x$ by $m$ samples within a circular buffer of size $N$.

* **Returns**: y (output sequence containing the circular shift)

### **Arithmetic & Systems**

#### **sigadd(x1, n1, x2, n2)**

Adds two signals $x_1[n]$ and $x_2[n]$ with different time supports.

* Automatically aligns time vectors and handles zero-padding.  
* **Returns:** y (summed signal), n (common time vector)

#### **sigmult(x1, n1, x2, n2)**

Multiplies two signals $x_1[n]$ and $x_2[n]$ element-wise.

* **Returns:** y (product signal), n (common time vector)

#### **conv_ext(x, nx, h, nh)**

Computes the convolution $y[n] = x[n] * h[n]$ with correct time index calculation.

* **Returns:** y (convolved signal), ny (resulting time vector)

#### **corr_ext(x, nx, h, nh)**

Computes the correlation $y[n] = x[n] $\star$ h[n]$ with correct time index calculation.

* **Returns:** y (convolved signal), ny (resulting time vector)

#### **circonvt(x1, x2, N)**

Computes the $N$-point circular convolution between two time-domain sequences $x_1$ and $x_2$.

* **Returns**: y (output sequence containing the circular convolution)

#### **hsolpsav(x, h, N)**

Implements the high-speed Overlap-Save method for block convolutions using FFT.

* **Parameters**: x (input sequence), h (impulse response), N (block length, must be a power of two).
* **Returns**: y (output convolution sequence)

### **Transforms**

#### **DTFT(x, n1, n2, k, delta)**

Computes the Discrete-Time Fourier Transform.

* **Parameters:** k (frequency index range), delta (frequency step size in radians).  
* **Returns:** w (frequency vector), X (computed DTFT values)

#### **approxCTFT(x, t\_start, t\_end, num\_points, w\_max, delta)**

Numerical approximation of the Continuous-Time Fourier Transform.

* **Returns:** t (time vector), x (sampled signal), W (freq vector), Xa (approximate CTFT)

#### **DFT(xn, N)**

Computes the Discrete Fourier Transform.

* **Returns:** Xk (DFT coeff. array over $0 \leq k \leq N-1$)

#### **IDFT(Xn, N)**

Computes the inverse Discrete Fourier Transform.

* **Returns:** xn (N-point sequence over $0 \leq n \leq N-1$)

#### **dfs(xn, N)**

Computes the Discrete Fourier Series (DFS) coefficients of one period of a periodic signal.

* **Returns**: Xk (DFS coeff. array over $0 \leq k \leq N-1$)

#### **idfs(Xk, N)**

Computes the Inverse Discrete Fourier Series (IDFS) to reconstruct one period of a periodic signal.

* **Returns**: xn (One period of periodic signal over $0 \leq n \leq N-1$)

### Filter Realization

#### **dir2cas(b, a)**

Converts a direct-form IIR/FIR filter into a cascade-form representation using second-order sections.

* **Returns**: b0 (gain coefficient), B (matrix of real numerator coefficients), A (matrix of real denominator coefficients)

#### **casfiltr(b0, B, A, x)**

Filters an input sequence $x$ using a cascade form realization of IIR and FIR filters.

* **Returns**: y (output sequence)

## **📝 Example**

```python
import numpy as np  
import matplotlib.pyplot as plt  
from sigfunctions import stepseq, sigadd

# 1. Generate two step sequences  
n1, x1 = stepseq(0, 10, 0)  
n2, x2 = stepseq(5, 15, 0)

# 2. Add them together (handles different time ranges automatically)  
y, n = sigadd(x1, n1, x2, n2)

# 3. Plot  
plt.stem(n, y)  
plt.title("Signal Addition Example")  
plt.xlabel("n")  
plt.ylabel("Amplitude")  
plt.show()
```


