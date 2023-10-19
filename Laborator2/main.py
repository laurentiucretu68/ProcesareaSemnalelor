import numpy as np
import matplotlib.pyplot as plt


def ex1(interval):
    amplitudine = 2
    frecventa = 100
    faza = 0

    def generare_sinusoidal(ts):
        return amplitudine * np.sin(2 * np.pi * frecventa * ts + np.pi/2)

    def generare_cosinusoidal(ts):
        return amplitudine * np.cos(2 * np.pi * frecventa * ts + faza)

    def desenare(x):
        plt.subplot(2, 1, 1)
        plt.plot(x, generare_sinusoidal(x), color='red')
        plt.title('Sinusiodal')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(x, generare_cosinusoidal(x), color='green')
        plt.title('Cosinusoidal')
        plt.grid()

        plt.show()

    desenare(interval)


def ex2(interval):
    amplitudine = 1
    frecventa = 100
    faza = np.array([0, np.pi/2, np.pi, 2*np.pi/3])
    colors = np.array(['red', 'blue', 'green', 'pink'])
    SNR = np.array([0.1, 1, 10, 100])

    def generare_sinusoidal(ts, faza):
        return amplitudine * np.sin(2 * np.pi * frecventa * ts + faza)

    def desenare(x):
        plt.grid()
        plt.title('Ex2')
        k = 0
        for f in faza:
            plt.plot(x, generare_sinusoidal(x, f), color=colors[k])
            k += 1
        plt.show()

    def adaugare_zgomot(x):
        z = np.random.normal(0, 1, len(x))
        for i, snr_val in enumerate(SNR):
            gama = np.sqrt(np.linalg.norm(generare_sinusoidal(x, faza[i])) ** 2 / (snr_val * np.linalg.norm(z) ** 2))
            semnal_zgomot = generare_sinusoidal(x, faza[i]) + gama * z
            plt.subplot(4, 1, i+1)
            plt.plot(semnal_zgomot)
            plt.grid()

        plt.show()

    desenare(interval)
    adaugare_zgomot(interval)


if __name__ == '__main__':
    interval = np.linspace(start=0, stop=6, num=200)
    ex1(interval)
    ex2(interval)

