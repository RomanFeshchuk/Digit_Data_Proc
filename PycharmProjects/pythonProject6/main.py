import numpy as np
from matplotlib import pyplot as plt


matrix = np.array([[1,  2,  3,  4],
                   [5,  6,  7,  8],
                   [9, 10, 11, 12]])


slice_matrix = np.array(matrix[:2, 1:3])
print(slice_matrix)


def base_settings():
    plt.xlim(-np.pi, np.pi)
    plt.legend(loc='upper left', fontsize=11)
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'], fontsize=10)


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