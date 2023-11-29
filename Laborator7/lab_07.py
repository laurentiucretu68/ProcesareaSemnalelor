from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfreqz, sosfilt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral


def ex_1():
    def show_spectogram_log(x):
        y = np.fft.fftshift(np.fft.fft2(x))
        freq_db = 20 * np.log10(abs(y + 1e-20))
        img = axs[1].imshow(freq_db + 1e-15)
        fig.colorbar(img, ax=axs[1])
        axs[0].set_title("Semnalul")
        axs[1].set_title("Spectrul semnalului")
        plt.tight_layout()
        plt.show()

    n = 512
    n1, n2 = np.meshgrid(range(n), range(n))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("np.sin(2πn1 + 3πn2)")
    x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
    img = axs[0].imshow(x1)
    fig.colorbar(img, ax=axs[0])
    show_spectogram_log(x1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("np.sin(4πn1) + np.cos(6πn2)")
    x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
    img = axs[0].imshow(x2)
    fig.colorbar(img, ax=axs[0])
    show_spectogram_log(x2)

    def show_spectogram(y):
        img = axs[1].imshow(np.abs(np.fft.fftshift(y)) + 1e-15)
        fig.colorbar(img, ax=axs[1])
        axs[0].set_title("Semnalul")
        axs[1].set_title("Spectrul semnalului")
        plt.tight_layout()
        plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_0,5 - Y_0,N-5, altfel Y_m1,m2 = 0, oricare m1, m2")
    n = 50
    y1 = np.zeros((n, n))
    y1[0][5] = 1
    y1[0][n - 5] = 1
    img = axs[0].imshow(np.fft.ifft2(y1).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_5,0 - Y_N-5,0, altfel Y_m1,m2 = 0, oricare m1, m2")
    y2 = np.zeros((n, n))
    y2[5][0] = 1
    y2[n - 5][0] = 1
    img = axs[0].imshow(np.fft.ifft2(y2).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Y_5,5 - Y_N-5,N-5, altfel Y_m1,m2 = 0, oricare m1, m2")
    y3 = np.zeros((n, n))
    y3[5][5] = 1
    y3[n - 5][n - 5] = 1
    img = axs[0].imshow(np.fft.ifft2(y3).real)
    fig.colorbar(img, ax=axs[0])
    show_spectogram(y3)


def ex_2():
    x = misc.face(gray=True)
    x_freq = np.fft.fft2(x)
    prag_snr = 0.001
    x_freq_compressed = x_freq * (np.abs(x_freq) >= prag_snr * np.max(np.abs(x_freq)))
    x_compressed = np.fft.ifft2(x_freq_compressed).real

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Imaginea originala")
    axs[0].imshow(x, cmap='gray')
    freq_db = 20 * np.log10(abs(x_freq + 1e-20))
    axs[1].imshow(np.fft.fftshift(freq_db), cmap='gray')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Imaginea comprimata")
    axs[0].imshow(x_compressed, cmap='gray')
    freq_db = 20 * np.log10(abs(x_freq_compressed + 1e-20))
    axs[1].imshow(np.fft.fftshift(freq_db), cmap='gray')
    plt.tight_layout()
    plt.show()

    signal_energy = np.sum(np.abs(x) ** 2)
    noise_energy = np.sum(np.abs(x - x_compressed) ** 2)

    snr = signal_energy / noise_energy
    print(f"Raportul SNR dupa comprimare: {snr:.4f}")


def ex_3():
    x = misc.face(gray=True)
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=x.shape)
    x_noisy = x + noise

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(x, cmap='gray')
    axs[0].set_title("Imaginea initiala")
    axs[1].imshow(x_noisy, cmap='gray')
    axs[1].set_title("Dupa adaugarea zgomotului")

    x_denoised = gaussian_filter(x_noisy, sigma=3)

    axs[2].imshow(x_denoised, cmap='gray')
    axs[2].set_title("Filtru Gaussian (sigma=3)")
    plt.tight_layout()
    plt.show()

    snr_noisy = np.sum(np.abs(x) ** 2) / np.sum(np.abs(x - x_noisy) ** 2)
    snr_denoised = np.sum(np.abs(x) ** 2) / np.sum(np.abs(x - x_denoised) ** 2)

    print(f"Raportul SNR adaugarea zgomotului: {snr_noisy:.4f}")
    print(f"Raportul SNR dupa eliminarea zgomotului: {snr_denoised:.4f}")


def ex_4():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import butter, sosfreqz, sosfilt

    def butter_bandstop(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def apply_bandstop_filter(signal, lowcut, highcut, fs, order=4):
        sos = butter_bandstop(lowcut, highcut, fs, order=order)
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal

    def plot_spectrogram(file_path):
        # Încărcați fisierul audio
        fs, signal = wavfile.read(file_path)

        # Afișați spectrograma
        plt.figure(figsize=(10, 4))
        _, _, Sxx, _ = plt.specgram(signal, Fs=fs, cmap='viridis', aspect='auto')
        plt.title('Spectrogram al Instrumentului')
        plt.xlabel('Timp (secunde)')
        plt.ylabel('Frecvență (Hz)')
        plt.colorbar(label='Intensitate (dB)')
        plt.show()

        return Sxx

    def remove_instrument_from_audio(audio_file, instrument_file, output_file, lowcut, highcut, start_time, end_time):
        # Încărcați fișierele audio
        fs_audio, audio_signal = wavfile.read(audio_file)
        fs_instrument, instrument_signal = wavfile.read(instrument_file)

        # Asigurați-vă că ambele semnale au aceeași lungime
        min_length = min(len(audio_signal), len(instrument_signal))
        audio_signal = audio_signal[:min_length]
        instrument_signal = instrument_signal[:min_length]

        # Aplicați filtrul pas-banda pentru a reduce frecvențele asociate instrumentului
        filtered_audio_signal = apply_bandstop_filter(audio_signal, lowcut, highcut, fs_audio)

        # Identificați intervalul de eșantioane corespunzător celor 10 secunde
        start_sample = int(start_time * fs_audio)
        end_sample = int(end_time * fs_audio)

        # Eliminați sunetul instrumentului din intervalul specificat
        filtered_audio_signal[start_sample:end_sample] -= instrument_signal[start_sample:end_sample]

        # Salvați rezultatul într-un nou fișier audio
        wavfile.write(output_file, fs_audio, filtered_audio_signal.astype(np.int16))

    # Specificați calea către fișierele audio
    audio_file_path = 'audio.wav'
    instrument_file_path = 'instrument.wav'
    output_file_path = 'audio_without_instrument.wav'

    # Afișați spectrograma pentru a identifica intervalul de timp și frecvențe
    spectrogram = plot_spectrogram(instrument_file_path)

    # Specificați intervalul de timp pentru cele 10 secunde
    start_time = 5  # Începutul intervalului în secunde
    end_time = 15  # Sfârșitul intervalului în secunde

    # Specificați intervalul de frecvențe pe baza analizei spectrogramei
    # Acesta este un exemplu simplificat, puteți ajusta aceste valori în funcție de analiza spectrogramei
    lowcut = 1000
    highcut = 2000

    # Eliminați sunetul instrumentului din fisierul audio
    remove_instrument_from_audio(audio_file_path, instrument_file_path, output_file_path, lowcut, highcut, start_time,
                                 end_time)


if __name__ == '__main__':
    # ex_1()
    # ex_2()  # nu cred ca e ok luat snr
    # ex_3()  # cum se elimina zgomotul? filtru sau scadere noise?
    ex_4()
