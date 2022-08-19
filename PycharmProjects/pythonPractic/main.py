import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew

a, b = 2, 7
x_uniform = np.random.uniform(a, b, 1000)
print("Среднее значение : ", np.mean(x_uniform))
# Среднее значение :  4.509041452382744

matrix = np.array([[1,  2,  3,  4],
                   [5,  6,  7,  8],
                   [9, 10, 11, 12]])
print("Дисперсия : ", np.var(x_uniform))
# Дисперсия :  2.16485272480643

print("Коэффициент асимметрии : ", skew(x_uniform))
# Коэффициент асимметрии:  0.005000189423357963

slice_matrix = np.array(matrix[:2, 1:3])
print(slice_matrix)
print("Коэффициент эксцесса : ", kurtosis(x_uniform))
# Коэффициент эксцесса:  -1.2761066178229146

plt.hist(x_uniform, 10, edgecolor='k')
count, bins, ignored = plt.hist(x_uniform, 10, edgecolor='k', density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

def base_settings():
    plt.xlim(-np.pi, np.pi)
    plt.legend(loc='upper left', fontsize=11)
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'], fontsize=10)
mu, sigma = 17, 18
x_n = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(x_n, 30, edgecolor='k', density = True)
# %%
plt.plot(bins, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.show()

x_r = np.random.rayleigh(18, 1000)
count, bins, ignored = plt.hist(x_r, 20, edgecolor='k', density=True)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.figure(1, figsize=(6, 4))
plt.title('Trigonometric functions', fontsize=12)
plt.ylim(C.min() * 1.1, C.max() * 1.1)
plt.yticks([-1, 0, 1], fontsize=10)
plt.tight_layout()

plt.subplot(211)
plt.title('Trigonometric functions', fontsize=12)
plt.plot(X, S, color='red', linewidth=2.0, linestyle='-', label='sin')
base_settings()

plt.subplot(212)
plt.plot(X, C, color='blue', linewidth=2.0, linestyle='-', label='cos')
base_settings()


plt.show()
x_p = np.random.poisson(18, 1000)
count, bins, ignored = plt.hist(x_p, 20, edgecolor='k', density=True)
