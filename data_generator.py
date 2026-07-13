import numpy as np
import os  # For file loading
from scipy import signal
from typing import List, Tuple, Union

class DataGenerator:
    """
    Handles generation of L-LTF signals, channels, and transmission.
    Supports synthetic and measurement-based data.
    """
    def __init__(self, channel_width_mhz: int = 20, snr_db: float = 40.0, realistic_lltf: bool = True,
                 freq_offset_hz: float = 0.0, phase_noise_std: float = 0.0, timing_jitter_std: float = 0.0,
                 iq_gain_imbalance: float = 0.0, iq_phase_imbalance_deg: float = 0.0):
        self.channel_width_mhz = channel_width_mhz
        self.snr_db = snr_db
        self.realistic_lltf = realistic_lltf  # Enable realism enhancements
        # Hardware imperfection parameters
        self.freq_offset_hz = freq_offset_hz
        self.phase_noise_std = phase_noise_std
        self.timing_jitter_std = timing_jitter_std  # in fractional samples (std)
        self.iq_gain_imbalance = iq_gain_imbalance  # relative gain imbalance (linear, e.g. 0.01)
        self.iq_phase_imbalance_deg = iq_phase_imbalance_deg  # degrees
        
        # Set parameters based on IEEE 802.11n standards
        if channel_width_mhz == 20:
            self.fft_size = 64
            self.cp_len = 16
            self.sample_rate = 20e6
            self.active_subcarriers = 52
        elif channel_width_mhz == 40:
            self.fft_size = 128
            self.cp_len = 32
            self.sample_rate = 40e6
            self.active_subcarriers = 114
        else:
            raise ValueError("Channel width must be 20 or 40 MHz")
        
        self.ts = 1.0 / self.sample_rate
        self.subcarrier_spacing = self.sample_rate / self.fft_size
        
        # Generate L-LTF frequency domain pattern
        self.l_ltf_freq = self._generate_lltf_frequency_domain()
    
    def _generate_lltf_frequency_domain(self) -> np.ndarray:
        """Generate WiFi L-LTF frequency domain pattern (enhanced for realism)."""
        l_ltf_freq = np.zeros(self.fft_size, dtype=complex)
        
        if self.channel_width_mhz == 20:
            # Canonical 52-tone L-LTF (BPSK) for 20 MHz
            pos = np.array([
                1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1,
                -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1
            ], dtype=complex)
            neg = pos.copy()
            
            # Place into FFT bins
            l_ltf_freq[1:27] = pos
            l_ltf_freq[38:64] = neg
            
            if self.realistic_lltf:
                # Add realism: phase rotations and scaling for unit energy
                phase_rot = np.exp(1j * np.random.uniform(0, 2*np.pi, len(l_ltf_freq)))
                l_ltf_freq *= phase_rot
                l_ltf_freq /= np.sqrt(np.sum(np.abs(l_ltf_freq)**2))  # Normalize energy
        
        elif self.channel_width_mhz == 40:
            # Simplified 40 MHz pattern (placeholder)
            L_LTF_58 = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1,
                                 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1,
                                 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1], dtype=complex)
            l_ltf_freq[2:30] = L_LTF_58[0:28]
            l_ltf_freq[34:62] = L_LTF_58[28:56]
            l_ltf_freq[66:94] = L_LTF_58[0:28]
            l_ltf_freq[98:126] = L_LTF_58[28:56]
        
        return l_ltf_freq
    
    def generate_lltf_time_domain(self) -> np.ndarray:
        """Generate L-LTF in time domain with CP (enhanced for realism).
        Implements fractional timing jitter by applying a phase ramp in the frequency domain.
        Returns the time-domain L-LTF sequence with cyclic prefix.
        """
        # Optionally apply fractional timing jitter as a phase ramp across subcarriers
        ltf_freq = self.l_ltf_freq.copy()
        if self.realistic_lltf and self.timing_jitter_std > 0:
            # draw jitter in samples (can be fractional)
            delta_t_samples = np.random.randn() * self.timing_jitter_std
            delta_t = delta_t_samples * self.ts
            # frequency bins (in Hz) for direct FFT ordering
            freq_bins = np.fft.fftfreq(self.fft_size, d=self.ts)
            # apply phase ramp: multiply frequency-domain bins by exp(-j*2*pi*f*delta_t)
            ltf_freq = ltf_freq * np.exp(-1j * 2.0 * np.pi * freq_bins * delta_t)
        
        # time-domain long training symbol(s)
        l_ltf_one = np.fft.ifft(ltf_freq) * np.sqrt(self.fft_size)
        # Standard L-LTF consists of two identical long symbols; create both
        l_ltf_two = np.tile(l_ltf_one, 2)
        # Prepend cyclic prefix (last cp_len samples of the two-symbol sequence)
        l_ltf_with_cp = np.concatenate([l_ltf_two[-self.cp_len:], l_ltf_two])
        return l_ltf_with_cp
    
    def create_multipath_channel(self, delays_samples: List[Union[int, float]], 
                               amplitudes: List[complex]) -> np.ndarray:
        """Create multipath channel impulse response."""
        if len(delays_samples) == 0:
            return np.array([], dtype=complex)
        
        half_len = 20
        max_delay = int(np.floor(max(delays_samples))) + half_len + 1
        channel = np.zeros(max_delay, dtype=complex)
        
        for delay, amp in zip(delays_samples, amplitudes):
            n0 = int(np.floor(delay))
            idx = np.arange(n0 - half_len, n0 + half_len + 1)
            taps = np.sinc(idx - delay)
            window = np.hanning(len(taps))
            taps = taps * window
            valid = (idx >= 0) & (idx < len(channel))
            channel[idx[valid]] += amp * taps[valid]
        
        return channel
    
    def transmit_through_channel(self, tx_signal: np.ndarray, channel: np.ndarray) -> np.ndarray:
        """Transmit signal through channel with AWGN and realistic hardware imperfections.
        Applies linear convolution, then applies CFO (frequency offset), phase noise, and IQ imbalance before adding noise.
        """
        rx_signal = signal.fftconvolve(tx_signal, channel, mode='full')

        # Apply carrier frequency offset (CFO) across the received signal if specified
        N = len(rx_signal)
        if self.freq_offset_hz != 0.0 or self.phase_noise_std > 0.0:
            n = np.arange(N)
            # frequency offset term
            if self.freq_offset_hz != 0.0:
                cfo = np.exp(1j * 2.0 * np.pi * self.freq_offset_hz * n * self.ts)
            else:
                cfo = np.ones(N, dtype=complex)
            # phase noise: cumulative Wiener-like phase noise (small increments)
            if self.phase_noise_std > 0.0:
                increments = np.random.randn(N) * self.phase_noise_std
                phase_noise = np.cumsum(increments)
                ph_noise = np.exp(1j * phase_noise)
            else:
                ph_noise = np.ones(N, dtype=complex)
            rx_signal = rx_signal * cfo * ph_noise

        # Apply IQ imbalance if requested (simple gain & phase imbalance model)
        if abs(self.iq_gain_imbalance) > 0.0 or abs(self.iq_phase_imbalance_deg) > 0.0:
            g = self.iq_gain_imbalance
            phi = np.deg2rad(self.iq_phase_imbalance_deg)
            # apply gain imbalance and phase rotation between I and Q
            i_part = np.real(rx_signal) * (1.0 + g)
            q_part = np.imag(rx_signal) * (1.0 - g)
            rx_signal = (i_part + 1j * q_part) * np.exp(1j * phi)

        # Add AWGN
        signal_power = np.mean(np.abs(rx_signal)**2)
        noise_power = signal_power / (10**(self.snr_db / 10)) if signal_power > 0 else 0.0
        noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        return rx_signal + noise
    
    def load_measurement_data(self, file_path: str, data_type: str = 'cfr') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load measurement data from file.
        - data_type: 'cfr' (complex frequency response) or 'cir' (channel impulse response).
        Returns CFR or (CFR, freqs) for 'cfr', CIR for 'cir'.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Measurement file {file_path} not found.")
        
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.csv'):
            data = np.loadtxt(file_path, delimiter=',', dtype=complex)  # Assume complex CSV
        else:
            raise ValueError("Unsupported file format. Use .npy or .csv")
        
        if data_type == 'cfr':
            # Assume data is CFR; generate freqs if not provided
            freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, self.ts))
            return data, freqs
        elif data_type == 'cir':
            return data  # Impulse response
        else:
            raise ValueError("data_type must be 'cfr' or 'cir'")
