import os
import numpy as np
import matplotlib.pyplot as plt


def ex1():
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

    print("DFT este unitara?", np.iscomplexobj(F) and np.allclose(np.dot(F, np.conj(F).T), N * np.eye(N), atol=1e-20))


def ex2():
    def signal(time: np.ndarray, frecventa):
        return np.sin(2 * np.pi * frecventa * time)

    time = np.linspace(0, 1, 10**4)
    x = signal(time, 5)
    y = x * np.exp(-2j * np.pi * time)
    dist = np.abs(y)
    cmap = plt.get_cmap('inferno')

    if not os.path.exists('exercitiul2'):
        os.mkdir('exercitiul2')

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Exercitiul 2 - figura 1')
    axs[0].plot(time, x, color='green')
    axs[0].set_xlabel('Timp')
    axs[0].set_ylabel('Amplitudine')
    axs[0].grid()

    axs[1].scatter(np.real(y), np.imag(y), c=dist, cmap=cmap, s=1)
    axs[1].set_aspect('equal')
    axs[1].set_xlabel('Real')
    axs[1].set_ylabel('Imaginar')
    axs[1].grid()

    plot_filename = os.path.join('exercitiul2', f"exercitiul2_figura1.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

    omega = np.array([1, 2, 5, 7])
    plots = []
    colors = []
    for w in omega:
        z = x * np.exp(-2j * np.pi * time * w)
        dist = np.abs(z)
        colors.append(dist)
        plots.append(z)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Exercitiul 2 - figura 2')
    axs[0, 0].scatter(np.real(plots[0]), np.imag(plots[0]), c=colors[0], cmap=cmap, s=1)
    axs[0, 0].set_xlabel('Real')
    axs[0, 0].set_ylabel('Imaginar')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].grid()

    axs[0, 1].scatter(np.real(plots[1]), np.imag(plots[1]), c=colors[1], cmap=cmap, s=1)
    axs[0, 1].set_xlabel('Real')
    axs[0, 1].set_ylabel('Imaginar')
    axs[0, 1].set_aspect('equal')
    axs[0, 1].grid()

    axs[1, 0].scatter(np.real(plots[2]), np.imag(plots[2]), c=colors[2], cmap=cmap, s=1)
    axs[1, 0].set_xlabel('Real')
    axs[1, 0].set_ylabel('Imaginar')
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlim((-1, 1))
    axs[1, 0].set_ylim((-1, 1))
    axs[1, 0].grid()

    axs[1, 1].scatter(np.real(plots[3]), np.imag(plots[3]), c=colors[3], cmap=cmap, s=1)
    axs[1, 1].set_xlabel('Real')
    axs[1, 1].set_ylabel('Imaginar')
    axs[1, 1].set_aspect('equal')
    axs[1, 1].grid()

    plot_filename = os.path.join('exercitiul2', f"exercitiul2_figura2.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)


def ex3():
    def signal(time: np.ndarray, frecventa: float, amplitudine: float):
        return amplitudine * np.cos(2 * np.pi * frecventa * time)

    def dft(x: np.ndarray):
        n = x.shape[0]
        X = np.zeros(n, dtype=np.complex128)
        for m in range(n):
            for i in range(n):
                X[m] += x[i] * (np.cos(2 * np.pi * m * i/n) - 1j * np.sin(2 * np.pi * m * i/n))

        return X

    time = np.linspace(0, 1, 1000)
    frec = np.array([5, 20, 75])
    amplitudine = np.array([1, 2, 0.5])
    x = np.sum([signal(time, f, amplitudine[index]) for index, f in enumerate(frec)], axis=0)
    X = np.abs(dft(x))

    if not os.path.exists('exercitiul3'):
        os.mkdir('exercitiul3')

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Exercitiul 3')
    axs[0].plot(time, x)
    axs[0].set_xlabel('Timp (s)')
    axs[0].set_ylabel('x(t)')
    axs[0].grid()

    axs[1].stem(np.linspace(0, 100, 100), X[:100])
    axs[1].set_xlabel('Frecventa (Hz)')
    axs[1].set_ylabel('X(f)')
    axs[1].grid()

    plot_filename = os.path.join('exercitiul3', f"exercitiul3.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
