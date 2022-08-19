import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib.ticker import FuncFormatter

a, b = 2, 7
x_uniform = np.random.uniform(a, b, 1000)
print("Среднее значение : ", np.mean(x_uniform))
# Среднее значение :  4.509041452382744
SAMPLES = 3200 # количество отсчётов
t = np.linspace(0, 5, SAMPLES, endpoint=True) # массив отсчётов времени в интервале [0, 1]
garm_sign = np.cos(2 * np.pi * t) # тестовый сигнал

print("Дисперсия : ", np.var(x_uniform))
# Дисперсия :  2.16485272480643
array1 = np.zeros(3200, dtype=int)
array1[0:3200:40] = 1
array2 = np.zeros(3200, dtype=int)
array2[0:3200:50] = 1
array3 = np.zeros(3200, dtype=int)
array3[0:3200:100] = 1

print("Коэффициент асимметрии : ", skew(x_uniform))
# Коэффициент асимметрии:  0.005000189423357963
d_sign1 = garm_sign * array1
d_sign2 = garm_sign * array2
d_sign3 = garm_sign * array3

print("Коэффициент эксцесса : ", kurtosis(x_uniform))
# Коэффициент эксцесса:  -1.2761066178229146
plt.figure(figsize=(15, 10))
plt.plot(t, garm_sign, '-', linewidth=2.0, label='Непрерывный')
plt.step(t, d_sign1, 'k', linewidth=2.0)
plt.grid()
plt.xlim([0, 1])
plt.xlabel('Время (сек.)')
plt.ylabel('Амплитуда')
plt.legend()

plt.figure(figsize=(15, 10))
plt.plot(t, garm_sign, '-', linewidth=2.0, label='Непрерывный')
plt.step(t, d_sign2, 'k', linewidth=2.0)
plt.grid()
plt.xlim([0, 1])
plt.xlabel('Время (сек.)')
plt.ylabel('Амплитуда')
plt.legend()

plt.figure(figsize=(15, 10))
plt.plot(t, garm_sign, '-', linewidth=2.0, label='Непрерывный')
plt.step(t, d_sign3, 'k', linewidth=2.0)
plt.grid()
plt.xlim([0, 1])
plt.xlabel('Время (сек.)')
plt.ylabel('Амплитуда')
plt.legend()

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
def quantize_uniform(garm_sign_ampl, quant_min=-1.0, quant_max=1.0, quant_level=4):
    x_normalize = (garm_sign_ampl - quant_min) * (quant_level - 1) / (quant_max - quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.round(x_normalize)
    x_quant = x_normalize_quant * (quant_max - quant_min) / (quant_level - 1) + quant_min
    return x_quant

x_p = np.random.poisson(18, 1000)
count, bins, ignored = plt.hist(x_p, 20, edgecolor='k', density=True)
def plot_graph_quant_function(axis, quant_min=-1.0, quant_max=1.0, quant_level=256):
    x_cont = np.linspace(quant_min, quant_max, 1000)
    x_quant = quantize_uniform(x_cont, quant_min=quant_min,
                               quant_max=quant_max, quant_level=quant_level)
    quant_stepsize = (quant_max - quant_min) / (quant_level - 1)
    title = f'$L = {quant_level:d}, \\Delta = {quant_stepsize:.2f}$'
    # title = r'$L = %d, \Delta=%0.2f$' % (quant_level, quant_stepsize)
    error = np.abs(x_quant - x_cont)
    axis.plot(x_cont, x_cont, color='k', label='Исходная амплитуда')
    axis.plot(x_cont, x_quant, color='b', label='Квантованная амплитуда')
    axis.plot(x_cont, error, 'r--', label='Ошибка квантования')
    axis.set_title(title)
    axis.set_xlabel('Амплитуда')
    axis.set_ylabel('Квантованная амплитуда/ошибка')
    axis.set_xlim([quant_min, quant_max])
    axis.set_ylim([quant_min, quant_max])
    axis.grid('on')
    axis.legend()

plt.figure(figsize=(24, 6))
ax = plt.subplot(1, 4, 1)
plot_graph_quant_function(ax, quant_min=-1, quant_max=4, quant_level=8)
ax = plt.subplot(1, 4, 2)
plot_graph_quant_function(ax, quant_min=-2, quant_max=2, quant_level=16)
ax = plt.subplot(1, 4, 3)
plot_graph_quant_function(ax, quant_min=-1, quant_max=1, quant_level=32)
ax = plt.subplot(1, 4, 4)
plot_graph_quant_function(ax, quant_min=-1, quant_max=1, quant_level=64)
plt.tight_layout()
plt.show()
