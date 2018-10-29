# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
 
SAMPLE_RATE = 8192
N = 128 # Windowing
 
 
def fft(x):
    X = list()
    for k in range(0, N):
        window = 1 # np.sin(np.pi * (k+0.5)/N)**2
        X.append(np.complex(x[k] * window, 0))
 
    fft_rec(X)
    return X
 
def fft_rec(X):
    N = len(X)
 
    if N <= 1:
        return
 
    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])
 
    fft_rec(even)
    fft_rec(odd)
 
    for k in range(0, N//2):
        t = np.exp(np.complex(0, -2 * np.pi * k / N)) * odd[k]
        X[k] = even[k] + t
        X[N//2 + k] = even[k] - t
 
 
 
 
x_values = np.arange(0, N, 1)
 
x = np.sin((2*np.pi*x_values / 32.0)) # 32 - 256Hz
x += np.sin((2*np.pi*x_values / 8.0)) # 8 - 1024Hz
X = fft(x)
 
 
# Plotting 
_, plots = plt.subplots(2)
 
## Plot in time domain
plots[0].plot(x)
 
## Plot in frequent domain
powers_all = np.abs(np.divide(X, N//2))
powers = powers_all[0:N//2]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N//2)), N)
plots[1].plot(frequencies, powers)
#for xy in zip(frequencies, powers):      
plt.annotate('256', xy=(256, 1))    
plt.annotate('1024', xy=(1024, 1))
           
#next line can be used to see all the data being plotted to verify that correct frequencies are being resolved                       # <--
  # plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',color='blue') 
## Show plots
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()
