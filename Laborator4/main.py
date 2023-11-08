import numpy as np
import matplotlib.pyplot as plt
import time
import os


def ex1():
    def dft(x: np.ndarray):
        n = x.shape[0]
        X = np.zeros(n, dtype=np.complex128)
        for m in range(n):
            for i in range(n):
                X[m] += x[i] * np.e ** (-2j * np.pi * m * i / n)

        return X

    N = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    dft_times = []
    fft_times = []
    for n in N:
        t = np.linspace(0, 1, n)
        x = np.sin(2 * np.pi * 5 * t)

        start_dft = time.time()
        _ = np.abs(dft(x))
        end_dft = time.time()
        dft_times.append(end_dft - start_dft)

        start_fft = time.time()
        _ = np.abs(np.fft.fft(x))
        end_fft = time.time()
        fft_times.append(end_fft - start_fft)

    plt.plot(N, np.array(dft_times), label='DFT Time')
    plt.plot(N, np.array(fft_times), label='FFT Time')
    plt.xlabel('Dimensiunea semnalului')
    plt.ylabel('Timpul de rulare')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    if not os.path.exists('exercitiul1'):
        os.mkdir('exercitiul1')
    plot_filename = os.path.join('exercitiul1', f"exercitiul1.png")
    plt.savefig(plot_filename)
    plot_filename = os.path.join('exercitiul1', f"exercitiul1.pdf")
    plt.savefig(plot_filename)

    plt.show()


if __name__ == '__main__':
    ex1()

