import numpy as np
import matplotlib.pyplot as plt

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

