import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile


def ex1(time: np.ndarray):
    amplitudine = 2
    frecventa = 100

    def sin_signal(ts: np.ndarray):
        return amplitudine * np.sin(2 * np.pi * frecventa * ts + np.pi / 2)

    def cos_signal(ts: np.ndarray):
        return amplitudine * np.cos(2 * np.pi * frecventa * ts)

    def draw(time: np.ndarray):
        plt.subplot(2, 1, 1)
        plt.plot(time, sin_signal(time), color='red')
        plt.title('Sin Signal')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(time, cos_signal(time), color='green')
        plt.title('Cos Signal')
        plt.grid()

        plt.tight_layout()
        plt.show()

    draw(time)


def ex2(time: np.ndarray):
    amplitudine = 1
    frecventa = 100
    faza = np.array([0, np.pi / 2, np.pi, 2 * np.pi / 3])
    colors = np.array(['red', 'blue', 'green', 'pink'])
    SNR = np.array([0.1, 1, 10, 100])

    def sin_signal(ts: np.ndarray, faza: float):
        return amplitudine * np.sin(2 * np.pi * frecventa * ts + faza)

    def draw(time: np.ndarray):
        plt.grid()
        plt.title('Ex2')
        for index, f in enumerate(faza):
            plt.plot(time, sin_signal(time, f), color=colors[index])
        plt.show()

    def adaugare_zgomot(time: np.ndarray):
        z = np.random.normal(0, 1, len(time))
        for index, snr_val in enumerate(SNR):
            gama = np.sqrt(np.linalg.norm(sin_signal(time, faza[index])) ** 2 / (snr_val * np.linalg.norm(z) ** 2))
            semnal_zgomot = sin_signal(time, faza[0]) + gama * z

            plt.subplot(4, 1, index + 1)
            plt.plot(semnal_zgomot)
            plt.grid()

        plt.tight_layout()
        plt.show()

    draw(time)
    adaugare_zgomot(time)


def ex3():
    def pct_a():
        frecventa = 400
        nr_esantioane = 1600

        timp = np.linspace(0, 0.1, nr_esantioane)
        return np.sin(2 * np.pi * frecventa * timp)

    def pct_b():
        frecventa = 800
        durata = 3
        nr_esantioane = 10 ** 4

        timp = np.linspace(0, durata, nr_esantioane)
        return np.sin(2 * np.pi * frecventa * timp)

    def pct_c():
        frecventa = 240
        nr_esantioane = 10 ** 4

        timp = np.linspace(0, 0.1, nr_esantioane)
        return np.mod(frecventa * timp, 1)

    def pct_d():
        frecventa = 300
        nr_esantioane = 10 ** 4

        timp = np.linspace(0, 0.1, nr_esantioane)
        return np.sign(np.sin(2 * np.pi * frecventa * timp))

    def play_and_save_signal(titles: list):
        for index, x in enumerate(['a', 'b', 'c', 'd']):
            if x == 'a':
                signal = pct_a()
            elif x == 'b':
                signal = pct_b()
            elif x == 'c':
                signal = pct_c()
            else:
                signal = pct_d()

            wave = 44100
            sd.play(signal, wave)
            sd.wait()
            wavfile.write(f'{titles[index]}', wave, signal)

    def test_read_signal(title: str):
        sample_rate, audio_data = wavfile.read(title)
        sd.play(audio_data, sample_rate)
        sd.wait()

    titles = [
        'signal1.wav',
        'signal2.wav',
        'signal3.wav',
        'signal4.wav'
    ]
    play_and_save_signal(titles)
    test_read_signal(titles[0])


def ex4():
    def sin_signal(time: np.ndarray):
        frecventa = 400
        return np.sin(2 * np.pi * frecventa * time)

    def sqware_signal(time: np.ndarray):
        frecventa = 300
        return np.sign(np.sin(2 * np.pi * frecventa * time))

    def plot():
        time = np.linspace(start=0, stop=0.1, num=10 ** 4)
        signal1 = sin_signal(time)
        signal2 = sqware_signal(time)

        fig, axs = plt.subplots(3)
        axs[0].plot(time, signal1, color='red')
        axs[0].set_title('First signal')
        axs[0].grid(True)
        axs[0].set_xlim([0, 0.01])

        axs[1].plot(time, signal2, color='green')
        axs[1].set_title('Second signal')
        axs[1].grid(True)
        axs[1].set_xlim([0, 0.01])

        axs[2].plot(time, signal1 + signal2, color='blue')
        axs[2].set_title('Sum signal')
        axs[2].grid(True)
        axs[2].set_xlim([0, 0.01])

        plt.tight_layout()
        plt.show()

    plot()


def ex5():
    def signal(time: np.ndarray, frecventa: float):
        return np.sin(2 * np.pi * frecventa * time)

    time = np.linspace(0, 3, 10 ** 5)
    signal1 = signal(time, 200)
    signal2 = signal(time, 1000)
    signal3 = np.concatenate((signal1, signal2), axis=0)
    wave = 44100
    sd.play(signal3, wave)  # cu cat frecventa este mai mare cu atat sunetul este mai inalt
    sd.wait()


def ex6():
    def signal(time: np.ndarray, frecventa: float):
        return np.sin(2 * np.pi * frecventa * time)

    fs = 10 ** 4
    time = np.linspace(0, 0.1, fs)
    signal_1 = signal(time, fs / 2)     # vor fi 2 puncte pe perioada
    signal_2 = signal(time, fs / 4)     # vor fi 4 puncte pe perioada
    signal_3 = signal(time, 0)  # nu exista variatie a semnalului

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 6")
    axs[0].plot(time, signal_1, color='red')
    axs[0].set_title('First signal')
    axs[0].grid(True)
    axs[0].set_xlim([0, 0.002])

    axs[1].plot(time, signal_2, color='green')
    axs[1].set_title('Second signal')
    axs[1].grid(True)
    axs[1].set_xlim([0, 0.002])

    axs[2].plot(time, signal_3, color='blue')
    axs[2].set_title('Third signal')
    axs[2].grid(True)
    axs[2].set_xlim([0, 0.002])

    plt.tight_layout()
    plt.show()


def ex7():
    def signal(time: np.ndarray, frecventa: float):
        return np.sin(2 * np.pi * frecventa * time)

    initial_time = np.linspace(0, 0.5, 1000)
    initial_signal = signal(initial_time, 200)

    time_4 = initial_time[::4]
    signal_4 = initial_signal[::4]

    time_2 = initial_time[1::4]
    signal_2 = initial_signal[1::4]

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 7")
    axs[0].plot(initial_time, initial_signal, color='red')
    axs[0].set_title('First signal')
    axs[0].grid(True)
    axs[0].set_xlim([0, 0.007])

    axs[1].plot(time_4, signal_4, color='green')
    axs[1].set_title('Second signal')
    axs[1].grid(True)
    axs[1].set_xlim([0, 0.007])

    axs[2].plot(time_2, signal_2, color='blue')
    axs[2].set_title('Third signal')
    axs[2].grid(True)
    axs[2].set_xlim([0, 0.007])

    plt.tight_layout()
    plt.show()

    """
        Decimarea unui semnal implica pastrarea unui subset de esantioane din semnalul
        initial. Semnalele decimate au o frecventa de 4 ori mai mica decat semnalul initial
        ceea ce afecteaza forma lor. Al doilea semnal decimat incepe mai tarziu decat cel initial
        ceea ce va afecta si mai tare forma acestuia.
    """


def ex8():
    interval = np.linspace(-np.pi/2, np.pi/2, 10**3)
    signal_1 = np.sin(interval)
    signal_2 = interval
    signal_pade = (interval - 7 * interval ** 3 / 60) / (1 + interval ** 2 / 20)
    eroare = np.abs(signal_1 - signal_2)

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 8 - Taylor")
    axs[0].plot(interval, signal_1, color='red')
    axs[0].axhline(0, color='black', linewidth=1)
    axs[0].axvline(0, color='black', linewidth=1)
    axs[0].set_title('sin(alpha)')
    axs[0].grid(True)

    axs[1].plot(interval, signal_2, color='green')
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].axvline(0, color='black', linewidth=1)
    axs[1].set_title('alpha')
    axs[1].grid(True)

    axs[2].plot(interval, eroare, color='blue')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].axvline(0, color='black', linewidth=1)
    axs[2].set_title('error')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 8 - Pade")
    axs[0].plot(interval, signal_1, color='red')
    axs[0].axhline(0, color='black', linewidth=1)
    axs[0].axvline(0, color='black', linewidth=1)
    axs[0].set_title('sin(alpha)')
    axs[0].grid(True)

    axs[1].plot(interval, signal_pade, color='green')
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].axvline(0, color='black', linewidth=1)
    axs[1].set_title('alpha')
    axs[1].grid(True)

    axs[2].plot(interval, np.abs(signal_1 - signal_pade), color='blue')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].axvline(0, color='black', linewidth=1)
    axs[2].set_title('error')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 8 - Taylor Oy logaritmic")
    axs[0].plot(interval, signal_1, color='red')
    axs[0].axhline(0, color='black', linewidth=1)
    axs[0].axvline(0, color='black', linewidth=1)
    axs[0].set_title('sin(alpha)')
    axs[0].grid(True)

    axs[1].plot(interval, signal_2, color='green')
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].axvline(0, color='black', linewidth=1)
    axs[1].set_title('alpha')
    axs[1].grid(True)

    axs[2].plot(interval, eroare, color='blue')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].axvline(0, color='black', linewidth=1)
    axs[2].set_title('error')
    axs[2].grid(True)

    plt.tight_layout()
    plt.yscale('log')
    plt.show()

    fig, axs = plt.subplots(3)
    fig.suptitle("Exercitiul 8 - Pade Oy logaritmic")
    axs[0].plot(interval, signal_1, color='red')
    axs[0].axhline(0, color='black', linewidth=1)
    axs[0].axvline(0, color='black', linewidth=1)
    axs[0].set_title('sin(alpha)')
    axs[0].grid(True)

    axs[1].plot(interval, signal_pade, color='green')
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].axvline(0, color='black', linewidth=1)
    axs[1].set_title('alpha')
    axs[1].grid(True)

    axs[2].plot(interval, np.abs(signal_1 - signal_pade), color='blue')
    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].axvline(0, color='black', linewidth=1)
    axs[2].set_title('error')
    axs[2].grid(True)

    plt.tight_layout()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    time = np.linspace(start=0, stop=6, num=200)
    ex1(time)
    ex2(time)
    ex3()
    ex4()
    ex5()
    ex6()
    ex7()
    ex8()
