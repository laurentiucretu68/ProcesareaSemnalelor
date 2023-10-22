import numpy as np
import matplotlib.pyplot as plt

"""
Teorie:
    * semnal discret: functie x : Z → R
    * relatia semnal continuu - discretizat: x[n] = x(nT)
        ** T: perioada de esantionare
        ** frecventa de esantionare: fs = 1/T:
            *** masoara nr de esantioane pe sec
            *** se masoara in Hz
"""


def ex1():
    def signal1(t: np.numarray):
        return np.cos(520 * np.pi * t + np.pi / 3)

    def signal2(t: np.numarray):
        return np.cos(280 * np.pi * t - np.pi / 3)

    def signal3(t: np.numarray):
        return np.cos(120 * np.pi * t + np.pi / 3)

    def plot_signals(n: int, titles: list, values: np.numarray, time: np.ndarray, colors: list, sampled=False):
        fig, axs = plt.subplots(n)
        for index, value in enumerate(values):
            if sampled:
                axs[index].stem(time, value)
            axs[index].plot(time, value, color=colors[index], marker='o')
            axs[index].set_title(titles[index])

        plt.tight_layout()
        plt.show()

    initial_step = 0.0005
    time = np.arange(0, 0.03, initial_step)
    signals = np.array([
        signal1(time),
        signal2(time),
        signal3(time)
    ])
    titles = ['Signal 1', 'Signal 2', 'Signal 3']
    colors = ['red', 'green', 'cyan']
    plot_signals(3, titles, signals, time, colors)

    fs = 200
    T = 1 / fs
    time_sampled = np.arange(0, 0.03, T)
    signals_sampled = np.array([
        signal1(time_sampled),
        signal3(time_sampled),
        signal3(time_sampled)
    ])

    titles_sampled = ['Sampled Signal 1', 'Sampled Signal 2', 'Sampled Signal 3']
    plot_signals(3, titles_sampled, signals_sampled, time_sampled, colors, sampled=True)


def ex2():
    def plot_signal(timp: np.ndarray, semnal: np.ndarray, title: str, lim: list):
        plt.plot(timp, semnal)
        plt.xlim(lim)
        plt.title(title)
        plt.xlabel('Timp (secunde)')
        plt.ylabel('Amplitudine')
        plt.grid(True)
        plt.show()

    def pct_a():
        frecventa = 400
        nr_esantioane = 1600

        timp = np.linspace(0, 0.1, nr_esantioane)
        semnal = np.sin(2 * np.pi * frecventa * timp)
        plot_signal(timp, semnal, 'Semnal sinusoidal de 400 Hz', [0, 0.01])

    def pct_b():
        frecventa = 800
        durata = 3
        nr_esantioane = 10**6

        timp = np.linspace(0, durata, nr_esantioane)
        semnal = np.sin(2 * np.pi * frecventa * timp)
        plot_signal(timp, semnal, 'Semnal sinusoidal de 800 Hz', [0, 0.01])

    def pct_c():
        frecventa = 300
        nr_esantioane = 10**6

        timp = np.linspace(0, 0.1, nr_esantioane)
        semnal_sawtooth = np.mod(frecventa * timp, 1)
        print(semnal_sawtooth)
        plot_signal(timp, semnal_sawtooth, 'Semnal sawtooth de 240 Hz', [0, 0.01])

    def pct_d():
        frecventa = 300
        nr_esantioane = 10**6

        timp = np.linspace(0, 0.1, nr_esantioane)
        semnal_sqware = np.sign(np.sin(2 * np.pi * frecventa * timp))
        plot_signal(timp, semnal_sqware, 'Semnal sqware de 300 Hz', [0, 0.1])

    def pct_e():
        semnal_aleator = np.random.rand(128, 128)

        plt.imshow(semnal_aleator)
        plt.colorbar()
        plt.title("Semnal 2D Aleator")
        plt.show()

    def pct_f():
        semnal_personalizat = np.zeros((128, 128))
        semnal_personalizat[::2, 1::3] = 1

        plt.imshow(semnal_personalizat, cmap='gray')
        plt.title("Semnal 2D Personalizat")
        plt.show()

    pct_a()
    pct_b()
    pct_c()
    pct_d()
    pct_e()
    pct_f()


def ex3():
    def pct_a():
        f = 2000
        t = 1 / f
        print(t)

    def pct_b():
        f = 2000
        nr_biti_esantion = 4
        nr_esantioane_ora = f * 3600    # 3600 = nr de secunde intr-o ora
        nr_biti_ora = nr_esantioane_ora * nr_biti_esantion
        nr_bytes_ora = nr_biti_ora / 8
        print(nr_bytes_ora)

    pct_a()
    pct_b()


if __name__ == '__main__':
    # ex1()
    ex2()
    # ex3()
