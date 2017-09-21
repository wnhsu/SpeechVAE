import os
import time
import scipy
import numpy as np

def mel_scale(freq):
    return 1127.0 * np.log(1.0 + float(freq)/700)

def inv_mel_scale(mel_freq):
    return 700 * (np.exp(float(mel_freq)/1127) - 1)

class MelBank(object):
    def __init__(self, 
                 low_freq=20, 
                 high_freq=8000, 
                 num_bins=80, 
                 sample_freq=16000, 
                 frame_size=32):

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        self.sample_freq = sample_freq
        self.frame_size = frame_size

        # frame_size in millisecond
        self.window_size = self.sample_freq * 0.001 * self.frame_size
        self.fft_freqs = np.linspace(
                0, self.sample_freq / 2, self.window_size / 2 + 1)[:-1]

        self.mel_low_freq = mel_scale(self.low_freq)
        self.mel_high_freq = mel_scale(self.high_freq)
        
        mel_freqs = np.linspace(
                self.mel_low_freq, self.mel_high_freq, self.num_bins+2)
        self.mel_windows = [mel_freqs[i:i+3] for i in xrange(self.num_bins)]

        def _weight(mel_window, mel_freq):
            mel_low, mel_center, mel_high = mel_window
            if mel_freq > mel_low and mel_freq < mel_high:
                if mel_freq <= mel_center:
                    return (mel_freq - mel_low) / (mel_center - mel_low)
                else:
                    return (mel_high - mel_freq) / (mel_high - mel_center)
            else:
                return 0
            
        self.mel_banks = [[_weight(window, mel_scale(freq)) \
                for freq in self.fft_freqs] for window in self.mel_windows]
        self.center_freqs = [inv_mel_scale(mel_freq) \
                for mel_freq in mel_freqs[1:-1]]

def hann(n):
    """
    n   : length of the window
    """
    w=np.zeros(n)
    for x in xrange(n):
        w[x] = 0.5*(1 - np.cos(2*np.pi*x/n)) 	
    return w

def stft_index(wave, frame_size_n, frame_starts_n, fft_size=None, win=None):
    """
    wave            : 1-d float array
    frame_size_n    : number of samples in each frame
    frame_starts_n  : a list of int denoting starting sample index of each frame
    fft_size        : number of frequency bins
    win             : windowing function on amplitude; len(win) == frame_size_n
    """
    wave = np.asarray(wave)
    frame_starts_n = np.int32(frame_starts_n)
    if fft_size is None:
        fft_size = frame_size_n
    if win is None:
        win = np.sqrt(hann(frame_size_n))

    # sanity check
    if not wave.ndim == 1:
        raise ValueError('wave is not mono')
    elif not frame_starts_n.ndim == 1:
        raise ValueError('frame_starts_n is not 1-d')
    elif not len(win) == frame_size_n:
        raise ValueError('win does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif fft_size % 2 == 1:
        raise ValueError('odd ffts not yet implemented')
    elif np.min(frame_starts_n) < 0 or np.max(frame_starts_n) > wave.shape[0]-frame_size_n:
        raise ValueError('Your starting indices contain values outside the allowed range')

    spec = np.asarray([scipy.fft(wave[n:n+frame_size_n]*win, n=fft_size)[:fft_size/2+1] \
                       for n in frame_starts_n])
    return spec	
   
def comp_spec_image(wave, decom, frame_size_n, shift_size_n, fft_size, awin, log_floor):
    """
    RETURN: 
        float matrix of shape (2, T, F)
    """
    frame_starts_n = np.arange(0, wave.shape[0]-frame_size_n, step=shift_size_n)
    spec = stft_index(wave, frame_size_n, frame_starts_n, fft_size, awin)
      
    if decom == "mp":
        phase = np.angle(spec)
        dbmag = np.log10(np.absolute(spec))
        dbmag[dbmag < log_floor] = log_floor
        dbmag = 20 * dbmag
        spec_image = np.concatenate([dbmag[None,...], phase[None,...]], axis=0)
    elif decom == "ri":
        real = np.real(spec)
        imag = np.imag(spec)
        spec_image = np.concatenate([real[None,...], imag[None,...]], axis=0)
    else:
        raise ValueError("decomposition type %s not supported" % decom)

    return spec_image
