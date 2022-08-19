import numpy as np
import matplotlib.pyplot as plt

N = 128
x_rect = np.zeros(N)
x_rect[64:96] = 1
T = len(x_rect)
t = np.arange(T)
def cof(x_rect, N):
    An = []
    Bn = []
    for n in range(N+1):
        sum = 0; sum1 = 0
        for i in range(T):
            sum = sum + np.cos(2 * np.pi * n * i/T) * x_rect[i]
            sum1 = sum1 + np.sin(2 * np.pi * n * i/T) * x_rect[i]
        An.append(sum * (2/T))
        Bn.append(sum1 * (2/T))
    return An, Bn
def recon(AB, T):
    An = AB[0]
    Bn = AB[1]
    xt = np.zeros(T)
    for i in range(T):
        sum = 0
        for n in range(1, len(An)):
            sum = sum + An[n] * np.cos(2 * np.pi * i * n/T) + Bn[n] * np.sin(2 * np.pi * n * i/T)
        xt[i] = (An[0]/2 + sum)
    return xt
plt.plot(x_rect, label= 'init signal')
for i in [5, 20]:
    plt.plot(t,recon(cof(x_rect, i), T), label= f'{i} гармоник')
plt.legend()
plt.grid()
plt.show()