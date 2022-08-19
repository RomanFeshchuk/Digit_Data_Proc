import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew

a, b = 2, 7
x_uniform = np.random.uniform(a, b, 1000)
print("Среднее значение : ", np.mean(x_uniform))
# Среднее значение :  4.509041452382744

print("Дисперсия : ", np.var(x_uniform))
# Дисперсия :  2.16485272480643

print("Коэффициент асимметрии : ", skew(x_uniform))
# Коэффициент асимметрии:  0.005000189423357963

print("Коэффициент эксцесса : ", kurtosis(x_uniform))
# Коэффициент эксцесса:  -1.2761066178229146

plt.hist(x_uniform, 10, edgecolor='k')
count, bins, ignored = plt.hist(x_uniform, 10, edgecolor='k', density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

mu, sigma = 17, 18
x_n = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(x_n, 30, edgecolor='k', density = True)
# %%
plt.plot(bins, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.show()

x_r = np.random.rayleigh(18, 1000)
count, bins, ignored = plt.hist(x_r, 20, edgecolor='k', density=True)

x_p = np.random.poisson(18, 1000)
count, bins, ignored = plt.hist(x_p, 20, edgecolor='k', density=True)
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
