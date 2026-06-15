import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def impseq(n1,n2,n0):
    '''
        Call the function impseq by declaring:
            n1 (lower bound of time vector n)
            n2 (upper bound of time vector n)
            n0 (time shifting, + if e.g delta[n-2], - if delta[n+2])
        
        Return values:
            n (time vector n)
            x (discrete function x[n])
    '''
    # Generates $\delta[n-n0], n1 \leq n \leq n2$
    n = np.arange(n1,n2+1)
    x = np.where(n == n0, 1.0, 0.0)
    return n, x

def stepseq(n1,n2,n0):
    '''
        Call the function stepseq by declaring:
            n1 (lower bound of time vector n)
            n2 (upper bound of time vector n)
            n0 (time shifting, + if e.g u[n-2], - if u[n+2])
        
        Return values:
            n (time vector n)
            x (discrete function x[n])
    '''
    # Generates $u[n-n0], n1 \leq n \leq n2$
    n = np.arange(n1,n2+1)
    x = np.where(n - n0 >= 0, 1.0, 0.0)
    return n,x

def sigadd(x1,n1,x2,n2):
    '''
        Call the function sigadd by declaring:
            x1 (discrete function x_1[n])
            x2 (discrete function x_2[n])
            n1 (time vector of first function x_1[n])
            n2 (time vector of first function x_2[n])
        
        Return values:
            n (time vector n)
            y (output sequence y[n]=x1+x2)
    '''
    # Adds two signals given n1=min:max; n2 = min:max and two discrete signals x1, x2
    n = np.arange(min(n1.min(),n2.min()),max(n1.max(),n2.max())+1)
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))

    mask1 = (n>=n1.min()) & (n<=n1.max())
    y1[mask1]=x1
    
    mask2 = (n>=n2.min()) & (n<=n2.max())
    y2[mask2]=x2 

    y = y1 + y2
    return y, n

def sigmult(x1,n1,x2,n2):
    '''
        Call the function sigmult by declaring:
            x1 (discrete function x_1[n])
            x2 (discrete function x_2[n])
            n1 (time vector of first function x_1[n])
            n2 (time vector of first function x_2[n])
        
        Return values:
            n (time vector n)
            y (output sequence y[n]=x1*x2)
    '''
    # Multiplies two signals given n1=min:max; n2 = min:max and two discrete signals x1, x2
    n = np.arange(min(n1.min(),n2.min()),max(n1.max(),n2.max())+1)
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))

    mask1 = (n>=n1.min()) & (n<=n1.max())
    y1[mask1]=x1
    
    mask2 = (n>=n2.min()) & (n<=n2.max())
    y2[mask1]=x2 

    y = y1*y2
    return y, n

def sigshift(x,m,k):
    '''
        Call the function sigshift by declaring:
            x (discrete function x[n])
            m (shifting value)
            k (shifted by k units)
        
        Return values:
            n (time vector n)
            y (output sequence y[n]=x[n+m])
    '''
    n_new = m + k
    y_new = x.copy()
    return y_new, n_new

def sigfold(x,n):
    '''
        Call the function sigfold by declaring:
            x (discrete function x[n])
            n (time vector n)
        
        Return values:
            n_folded (folded time vector n_fold)
            y (output sequence y[n]=x[-n])
    '''
    # Folds a signal x[n], y[n]=x[-n]
    y = np.flip(x)
    n_folded = -np.flip(n)
    
    return n_folded, y

def evenodd(x,n):
    '''
        Call the function evenodd by declaring:
            x (discrete function x[n])
            n (time vector n)
        
        Return values:
            n (time vector n)
            y (output sequence y[n]=x[n+m])
    '''
    # Breaks the signal x[n] into even and odd components
    if np.any(np.imag(x) != 0):
        raise ValueError('x is not a real sequence')

    m_flipped = -n[::-1] 

    m1 = min(m_flipped.min(), n.min())
    m2 = max(m_flipped.max(), n.max())
    m = np.arange(m1, m2 + 1)

    nm = n[0] - m[0]
    
    x1 = np.zeros(len(m))
    
    x1[nm : nm + len(x)] = x
    x = x1
    
    xe = 0.5 * (x + x[::-1])
    xo = 0.5 * (x - x[::-1])
    
    return xe, xo, m

def conv_ext(x,nx,h,nh):
    '''
    Call the function conv_ext by declaring:
        nx (range of values from lower bound to upper bound of n for x[n])
        x (discrete function x[n])
        nh (range of values from lower bound to upper bound of n for h[n])
        h (discrete function x[n])

    Return values:
        y (output of convolution y[n] = x[n]*h[n])
        ny (output time vector)
    '''
    nyb = nx[0]+nh[0]
    nye = nx[len(x)-1]+nh[len(h)-1]
    ny = np.arange(nyb,nye+1)
    y = np.convolve(x,h)

    return y, ny

def corr_ext(x, nx, h, nh):
    '''
    Call the function corr_ext by declaring:
        nx (range of values from lower bound to upper bound of n for x[n])
        x (discrete function x[n])
        nh (range of values from lower bound to upper bound of n for h[n])
        h (discrete function h[n])

    Return values:
        y (output of correlation y[n] = x[n] star h[n])
        ny (output time vector)
    '''
    # Correlation lags are: (nx_start - nh_end) to (nx_end - nh_start)
    nyb = nx[0] - nh[-1]
    nye = nx[-1] - nh[0]
    ny = np.arange(nyb, nye + 1)
    
    # 2. Perform Correlation
    # We use 'full' mode to get the complete correlation result
    y = np.correlate(x, h, mode='full')
    
    return y, ny

def DTFT(x,n1,n2,k,delta):
    '''
    Call the function DTFT by declaring:
        x (discrete function x[n])
        n1 (lower bound of the time interval)
        n2 (upper bound of the time interval)
        k (range of frequency indices, e.g., 0 to 500)
        delta (frequency step size in radians)

    Return values:
        w (frequency vector)
        X (computed DTFT values)
    '''

    # Create the time interval array n
    n = np.arange(n1,n2+1)

    # Create the frequency array w
    w = k*delta

    # DTFT summation
    X = np.array([np.sum(x*np.exp(-1j*omega*n)) for omega in w])

    return w, X

def approxCTFT(x,t_start, t_end, num_points, w_max, delta):
    '''
        Call the function approxDTFT by declaring:
        dt (approximation of the dt from the CTFT integral)
        t (range of values starting from t_start to t_end, with step size equal to N; see number 3)
        num_points = int((t_end-t_start)/dt)+1
        w_max (the maximum frequency in rad/s to compute)
        K (the number of frequency intervals)

        Return values:
        t (generated time vector)
        x (sampled analog signal)
        W (frequency vector)
        Xa (computed approximation of CTFT)
    '''

    # Call the function by defining, Dt, t (range of values for time vector), function x(n)
    t = np.linspace(t_start, t_end, num_points)
    dt = T[1] - T[0]

    x = x_func(t)

    k = np.arange(0,delta+1)
    W = k*w_max/delta

    X = np.array([np.sum(x*np.exp(-1j*omega*n))*dt for omega in W])

    return t, x, W, X

def DFT(xn,N):
    '''
        Computes Discrete Fourier Transform using list comprehension.
    
    Call the function DFT by declaring:
        xn (N-point finite-duration sequence)
        N (Length of DFT)
        
    Return values:
        Xk (DFT coeff. array over 0 <= k <= N-1)
    '''

    # Create the time vector n
    n = np.arange(N)

    xn = np.array(xn)
    # If the length of the N-point finite sequence is less than N, pad 0's
    if len(xn) < N:
        xn = np.pad(xn, (0, N - len(xn)), 'constant')

    # Solve the DFT
    Xk = np.array([np.sum(xn*np.exp(-1j*2*np.pi*n*k/N)) for k in range(N)])

    return Xk

def IDFT(Xk,N):
    '''
    Call the function IDFT by declaring:
    Xk (DFT coeff. array over 0 <= k <= N-1)
    N (Length of DFT)
        
    Return values:
    xn (N-point sequence over 0 <= n <= N-1)
    '''

    # Create the time vector n
    k = np.arange(N)

    # Solve the IDFT
    xn = np.array([1/N*np.sum(Xk*np.exp(1j*2*np.pi*k*n/N)) for n in range(N)])

    return xn

def ditFFT(xn,N):
    '''
    Call the function ditFFT (DIT-FFT Radix 2) by declaring:
        xn (N-point finite-duration sequence)
        N (Length of FFT, must be a power of 2)

    Return values:
        Xk (FFT coefficient array over 0 <= k <= N-1)
    '''

    xn = np.array(xn)

    if len(xn) < N:
        xn = np.pad(xn, (0, N-len(xn)), 'constant')

    xn = xn[:N]

    if N <= 1:
        return xn

    if N % 2 != 0:
        raise ValueError("N must be a power for this DIT-FFT Radix-2 Algorithm")

    even = ditFFT(xn[0::2], N // 2)
    odd = ditFFT(xn[x::2], N // 2)

    k = np.arange(N // 2)

    twiddle = np.exp(-1j*2*np.pi*k/N)
    Xk = np.concatenate([even+twiddle*odd, even - twiddle*odd])

    return Xk

def dfs(xn, N):
    '''
    Computes Discrete Fourier Series Coefficients
    
    Call the function dfs by declaring:
        xn (One period of periodic signal over 0 <= n <= N-1)
        N (Fundamental period of xn)
        
    Return values:
        Xk (DFS coeff. array over 0 <= k <= N-1)
    '''
    # Create row vectors for n and k
    n = np.arange(0, N)
    k = np.arange(0, N)
    
    # Wn factor
    WN = np.exp(-1j * 2 * np.pi / N)
    
    # Creates an N by N matrix of nk values using outer product (n' * k in MATLAB)
    nk = np.outer(n, k)
    
    # DFS matrix
    WNnk = WN ** nk
    
    # Convert xn to a numpy array explicitly for matrix multiplication
    xn = np.array(xn)
    
    # Row vector for DFS coefficients (equivalent to xn * WNnk in MATLAB)
    Xk = np.dot(xn, WNnk)
    
    return Xk

def idfs(Xk, N):
    '''
    Computes Inverse Discrete Fourier Series
    
    Call the function idfs by declaring:
        Xk (DFS coeff. array over 0 <= k <= N-1)
        N (Fundamental period of Xk)
        
    Return values:
        xn (One period of periodic signal over 0 <= n <= N-1)
    '''
    # Create row vectors for n and k
    n = np.arange(0, N)
    k = np.arange(0, N)
    
    # Wn factor
    WN = np.exp(-1j * 2 * np.pi / N)
    
    # Creates an N by N matrix of nk values using outer product (n' * k in MATLAB)
    nk = np.outer(n, k)
    
    # IDFS matrix (using element-wise negative exponent)
    WNnk = WN ** (-nk)
    
    # Convert Xk to a numpy array explicitly for matrix multiplication
    Xk = np.array(Xk)
    
    # Row vector for IDFS values scaled by 1/N
    xn = np.dot(Xk, WNnk) / N
    
    return xn

def circevod(x):
    '''
    Signal decomposition into circular-even and circular-odd parts
    
    Call the function circevod by declaring:
        x (discrete function x[n])
        
    Return values:
        xec (circular-even part)
        xoc (circular-odd part)
    '''
    # Verify if the sequence is real
    if np.any(np.imag(x) != 0):
        raise ValueError('x is not a real sequence')
        
    # Vector length and time index array
    N = len(x)
    n = np.arange(0, N)
    
    # Convert x to a numpy array explicitly to handle vectorized math
    x = np.array(x)
    
    # Circular folding index sequence: x((-n))_N
    fold_idx = np.mod(-n, N)
    
    # Compute circular-even and circular-odd sequences
    xec = 0.5 * (x + x[fold_idx])
    xoc = 0.5 * (x - x[fold_idx])
    
    return xec, xoc

def cirshftt(x, m, N):
    '''
    Circular shift of m samples wrt size N in sequence x: (time domain)
    
    Call the function cirshftt by declaring:
        x (input sequence of length <= N)
        m (sample shift)
        N (size of circular buffer)
        
    Return values:
        y (output sequence containing the circular shift)
    '''
    # Check for length of x
    if len(x) > N:
        raise ValueError('N must be >= the length of x')
        
    # Convert x to a numpy array explicitly
    x = np.array(x)
    
    # If the length of the finite sequence is less than N, pad with zeros
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)), 'constant')
        
    # Create time index array over 0 <= n <= N-1
    n = np.arange(0, N)
    
    # Method: y(n) = x((n-m) mod N)
    shift_idx = np.mod(n - m, N)
    y = x[shift_idx]
    
    return y

def circonvt(x1, x2, N):
    '''
    N-point circular convolution between x1 and x2: (time-domain)
    
    Call the function circonvt by declaring:
        x1 (input sequence of length N1 <= N)
        x2 (input sequence of length N2 <= N)
        N (size of circular buffer)
        
    Return values:
        y (output sequence containing the circular convolution)
    '''
    # Check for length of x1
    if len(x1) > N:
        raise ValueError('N must be >= the length of x1')
        
    # Check for length of x2
    if len(x2) > N:
        raise ValueError('N must be >= the length of x2')
        
    # Convert inputs to numpy arrays explicitly
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # Pad sequences with zeros if their lengths are less than N
    if len(x1) < N:
        x1 = np.pad(x1, (0, N - len(x1)), 'constant')
    if len(x2) < N:
        x2 = np.pad(x2, (0, N - len(x2)), 'constant')
        
    # Establish time index array over 0 <= m <= N-1
    m = np.arange(0, N)
    
    # Circularly fold x2 sequence: x2((-m) mod N)
    fold_idx = np.mod(-m, N)
    x2 = x2[fold_idx]
    
    # Initialize the transformation matrix H
    H = np.zeros((N, N))
    
    # Populate matrix rows using our cirshftt function
    for n in range(0, N):
        H[n, :] = cirshftt(x2, n, N)
        
    # Calculate convolution using row vector matrix multiplication (x1 * H')
    y = np.dot(x1, H.T)
    
    return y

def hsolpsav(x, h, N):
    '''
    High-speed Overlap-Save method of block convolutions using FFT
    
    Call the function hsolpsav by declaring:
        x (input sequence)
        h (impulse response)
        N (block length, will be automatically forced/checked as a power of two)
        
    Return values:
        y (output convolution sequence)
    '''
    # Ensure block length N is a power of two
    N = int(2**np.ceil(np.log2(N)))
    
    Lenx = len(x)
    M = len(h)
    M1 = M - 1
    L = N - M1
    
    # Compute the N-point DFT of the filter impulse response
    # (Using your library's native DFT function or standard np.fft.fft)
    h_fft = np.fft.fft(h, N)
    
    # Explicitly convert x to a numpy array for vector manipulations
    x = np.array(x)
    
    # Pre-append M1 zeros and post-append N-1 zeros to frame the input signal
    x = np.concatenate([np.zeros(M1), x, np.zeros(N - 1)])
    
    # Calculate the number of blocks needed
    K = int(np.floor((Lenx + M1 - 1) / L))
    
    # Initialize the block storage matrix Y with dimensions (K + 1, N)
    Y = np.zeros((K + 1, N))
    
    # Block processing loop
    for k in range(0, K + 1):
        # Extract an N-point block from the padded input array
        xk = np.fft.fft(x[k*L : k*L + N])
        
        # Circular convolution via element-wise product followed by IDFT
        Y[k, :] = np.real(np.fft.ifft(xk * h_fft))
        
    # Discard the first M1 (corrupted) elements of each block 
    # and flatten the valid regions into a single 1D row vector output
    Y_valid = Y[:, M1:N]
    y = Y_valid.flatten()
    
    return y

def dir2cas(b, a):
    '''
    DIRECT-form to CASCADE-form conversion
    
    Call the function dir2cas by declaring:
        b (numerator polynomial coefficients of DIRECT form)
        a (denominator polynomial coefficients of DIRECT form)
        
    Return values:
        b0 (gain coefficient)
        B (K by 3 matrix of real coefficients containing bk's)
        A (K by 3 matrix of real coefficients containing ak's)
    '''
    # Compute gain coefficient b0
    b0 = b[0]
    b = np.array(b, dtype=complex) / b0
    
    a0 = a[0]
    a = np.array(a, dtype=complex) / a0
    
    b0 = b0 / a0
    
    M = len(b)
    N = len(a)
    
    # Equalize lengths of polynomial arrays
    if N > M:
        b = np.pad(b, (0, N - M), 'constant')
    elif M > N:
        a = np.pad(a, (0, M - N), 'constant')
        N = M

    K = N // 2
    B = np.zeros((K, 3))
    A = np.zeros((K, 3))
    
    if K * 2 == N:
        b = np.append(b, 0.0)
        a = np.append(a, 0.0)
        
    # Helper lambda to replicate MATLAB's cplxpair behavior 
    # (groups complex conjugate pairs together, real numbers at the end)
    def cplxpair_python(roots):
        # Sort primarily by real part, secondarily by imaginary part
        return roots[np.lexsort((np.imag(roots), np.real(roots)))]

    broots = cplxpair_python(np.roots(b))
    aroots = cplxpair_python(np.roots(a))
    
    # Group roots pairwise into second-order sections
    for i in range(0, 2 * K, 2):
        Brow = broots[i : i + 2]
        # np.poly generates polynomial coefficients from roots
        Brow_poly = np.real(np.poly(Brow))
        B[i // 2, :] = Brow_poly
        
        Arow = aroots[i : i + 2]
        Arow_poly = np.real(np.poly(Arow))
        A[i // 2, :] = Arow_poly
        
    return b0, B, A


def casfiltr(b0, B, A, x):
    '''
    CASCADE form realization of IIR and FIR filters
    
    Call the function casfiltr by declaring:
        b0 (gain coefficient of CASCADE form)
        B (K by 3 matrix of real coefficients containing bk's)
        A (K by 3 matrix of real coefficients containing ak's)
        x (input sequence)
        
    Return values:
        y (output sequence)
    '''
    K, L = B.shape
    N = len(x)
    
    # w matrix tracks intermediary filtering steps across rows
    w = np.zeros((K + 1, N))
    w[0, :] = x
    
    # Pass the signal through each second-order section consecutively
    for i in range(0, K):
        w[i + 1, :] = lfilter(B[i, :], A[i, :], w[i, :])
        
    y = b0 * w[K, :]
    return y