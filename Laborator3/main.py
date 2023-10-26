import os
import numpy as np
import matplotlib.pyplot as plt


def ex1():
    def signal(time: np.ndarray, frecventa):
        return np.sin(2 * np.pi * frecventa * time)

    def dft(x: np.ndarray):
        X = np.zeros(x.shape[0], dtype=np.complex128)
        n = x.shape[0]
        for m in range(n):
            for i in range(n):
                X[m] += x[i] * (np.cos(2 * np.pi * m * m/n) - 1j * np.sin(2 * np.pi * m * m/n))

        return X

    def dft_matrix(n):
        F = np.zeros((n, n), dtype=np.complex128)

        for m in range(n):
            for i in range(n):
                F[m, i] = np.exp(2j * np.pi * m * i/n)

        return F

    def plot(F: np.ndarray, directory_name: str):
        n = F.shape[0]
        interval = np.linspace(0, n-1, n)

        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

        fig, axs = plt.subplots(n, 2, figsize=(16, 10))
        fig.suptitle('Exercitiul 1')
        for i in range(n):
            axs[i, 0].plot(interval, np.real(F[i]), color='red')
            axs[i, 1].plot(interval, np.imag(F[i]), color='green', linestyle='dashed')
            axs[i, 0].grid()
            axs[i, 1].grid()

        plot_filename = os.path.join(directory_name, f"exercitiul1.png")
        plt.savefig(plot_filename)
        plt.show()
        plt.close(fig)

    N = 8
    F = dft_matrix(N)
    plot(F, 'exercitiul1')

    print("Matricea Fourier este unitara?", np.iscomplexobj(F)
          and np.allclose(np.dot(F, np.conj(F).T), N * np.eye(N), atol=1e-10))


if __name__ == '__main__':
    ex1()
