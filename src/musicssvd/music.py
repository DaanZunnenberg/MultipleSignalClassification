import numpy as np
from .generator import DataGenerator
from .processor import DataProcessor
from .evaluator import Evaluator
from .plotter import Plotter

class MUSICTDE:
    """
    MUSIC-based Time Delay Estimation for IEEE 802.11n WiFi L-LTF
    Implements superresolution delay estimation using subspace methods
    with frequency smoothing for enhanced performance.
    """
    
    def __init__(self, channel_width_mhz: int = 20, snr_db: float = 40.0,
                 realistic_lltf: bool = True,
                 freq_offset_hz: float = 0.0,
                 phase_noise_std: float = 0.0,
                 timing_jitter_std: float = 0.0,
                 iq_gain_imbalance: float = 0.0,
                 iq_phase_imbalance_deg: float = 0.0):
        """
        Initialize MUSIC TDE for WiFi OFDM
        
        Parameters:
        channel_width_mhz: Channel bandwidth (20 or 40 MHz)
        snr_db: Signal-to-Noise Ratio in dB  
        """
        self.channel_width_mhz = channel_width_mhz
        self.snr_db = snr_db
        
        # store imperfection parameters
        self.realistic_lltf = realistic_lltf
        self.freq_offset_hz = freq_offset_hz
        self.phase_noise_std = phase_noise_std
        self.timing_jitter_std = timing_jitter_std
        self.iq_gain_imbalance = iq_gain_imbalance
        self.iq_phase_imbalance_deg = iq_phase_imbalance_deg

        # Create DataGenerator with the specified imperfections
        self.data_gen = DataGenerator(channel_width_mhz=channel_width_mhz,
                                      snr_db=snr_db,
                                      realistic_lltf=realistic_lltf,
                                      freq_offset_hz=freq_offset_hz,
                                      phase_noise_std=phase_noise_std,
                                      timing_jitter_std=timing_jitter_std,
                                      iq_gain_imbalance=iq_gain_imbalance,
                                      iq_phase_imbalance_deg=iq_phase_imbalance_deg)
        self.data_processor = DataProcessor(self.data_gen)
        self.evaluator = Evaluator()
        self.plotter = Plotter()
        
        # Expose some parameters for convenience
        self.fft_size = self.data_gen.fft_size
        self.cp_len = self.data_gen.cp_len
        self.ts = self.data_gen.ts
        self.sample_rate = self.data_gen.sample_rate
    
    def generate_lltf_time_domain(self):
        return self.data_gen.generate_lltf_time_domain()
    
    def create_multipath_channel(self, delays_samples, amplitudes):
        return self.data_gen.create_multipath_channel(delays_samples, amplitudes)
    
    def transmit_through_channel(self, tx_signal, channel, add_noise: bool = True, **overrides):
        return self.data_gen.transmit_through_channel(tx_signal, channel, add_noise=add_noise, **overrides)
    
    def cross_correlation_tde(self, rx_signal, search_range_samples=(0, 50), threshold=0.3):
        return self.data_processor.cross_correlation_tde(rx_signal, search_range_samples, threshold)
    
    def music_tde(self, rx_signal, num_paths, search_range_samples=(0, 50), subarray_size=None, non_overlapping=False):
        return self.data_processor.music_tde(rx_signal, num_paths, search_range_samples, subarray_size, non_overlapping)
    
    def esprit_tde(self, rx_signal, num_paths, search_range_samples=(0,50), subarray_size=None, non_overlapping=False, tau_grid_step=0.01):
        """ESPRIT implemented directly in MUSICTDE as a fallback if DataProcessor lacks implementation.
        Returns (tau_grid, pseudospectrum, estimated_delays)
        """
        import numpy as np
        from scipy import signal

        # extract single OFDM symbol (remove CP)
        rx_symbol = rx_signal[self.cp_len:self.cp_len + self.fft_size]
        rx_freq = np.fft.fft(rx_symbol) / np.sqrt(self.fft_size)
        rx_freq_shifted = np.fft.fftshift(rx_freq)
        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)

        used_mask = ltf_freq_shifted != 0
        H_estimated = rx_freq_shifted[used_mask] / ltf_freq_shifted[used_mask]
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.fft_size, self.ts))
        freqs_used = freqs_shifted[used_mask]

        if subarray_size is None:
            subarray_size = len(H_estimated) // 2

        # Build snapshots (overlapping forward/backward)
        N = len(H_estimated)
        L = subarray_size
        if L >= N:
            # fallback: single snapshot
            X = H_estimated.reshape(-1, 1)
            R = X @ X.conj().T
            freqs_subarray = freqs_used.copy()[:L]
        else:
            snapshots = []
            num_subarrays = N - L + 1
            for i in range(num_subarrays):
                snapshots.append(H_estimated[i:i+L])
            # forward-backward
            H_rev = np.conj(H_estimated[::-1])
            for i in range(num_subarrays):
                snapshots.append(H_rev[i:i+L])
            X = np.array(snapshots).T  # L x num_snapshots
            R = (X @ X.conj().T) / X.shape[1]
            # choose central subarray freqs
            start_idx = max(0, (N - L)//2)
            freqs_subarray = freqs_used[start_idx:start_idx+L]

        # EVD
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        if num_paths <= eigvecs.shape[1]:
            U_s = eigvecs[:, :num_paths]
        else:
            U_s = eigvecs

        if U_s.shape[0] < 2:
            return np.array([]), np.array([]), []

        U1 = U_s[:-1, :]
        U2 = U_s[1:, :]

        try:
            Phi = np.linalg.pinv(U1) @ U2
            eigvals_phi, _ = np.linalg.eig(Phi)
        except Exception:
            eigvals_phi = np.array([])

        # Map eigenvalues to delays
        taus_samples = np.array([])
        if len(freqs_subarray) > 1 and len(eigvals_phi) > 0:
            df = np.mean(np.diff(freqs_subarray))
            taus_sec_raw = -np.angle(eigvals_phi) / (2.0 * np.pi * df)
            period = 1.0 / df if df != 0 else np.inf
            start_s = search_range_samples[0] * self.ts
            end_s = search_range_samples[1] * self.ts
            center = 0.5 * (start_s + end_s)

            # shift each raw candidate toward center (principal wrap)
            taus_sec = np.zeros_like(taus_sec_raw, dtype=float)
            for ii, t in enumerate(taus_sec_raw):
                if np.isfinite(period) and period > 0:
                    n = np.round((center - t) / period)
                    taus_sec[ii] = t + n * period
                else:
                    taus_sec[ii] = t
            taus_samples = np.real(taus_sec / self.ts)

            # Cluster near-duplicate taus (caused by numerical splitting) -> keep distinct cluster centers
            if taus_samples.size > 0:
                sorted_idx = np.argsort(taus_samples)
                sorted_t = taus_samples[sorted_idx]
                clusters = []
                current = [sorted_t[0]]
                min_sep = 0.25  # samples: treat closer-than-this as duplicate
                for val in sorted_t[1:]:
                    if abs(val - current[-1]) <= min_sep:
                        current.append(val)
                    else:
                        clusters.append(np.mean(current))
                        current = [val]
                clusters.append(np.mean(current))
                taus_samples = np.array(clusters)

        # taus_samples may still be fewer than num_paths; build pseudospectrum and supplement if needed
        start_sample, end_sample = search_range_samples
        tau_grid = np.arange(start_sample, end_sample, tau_grid_step)
        # build pseudospectrum using noise subspace (for potential supplemental peaks)
        eigenvalues_full, eigenvectors_full = np.linalg.eigh(R)
        idx_full = np.argsort(eigenvalues_full)[::-1]
        eigenvectors_full = eigenvectors_full[:, idx_full]
        if eigenvectors_full.shape[1] > num_paths:
            noise_subspace = eigenvectors_full[:, num_paths:]
        else:
            noise_subspace = eigenvectors_full[:, 1:]

        spectrum = np.zeros(len(tau_grid))
        for i, tau_samp in enumerate(tau_grid):
            tau_sec = tau_samp * self.ts
            a = np.exp(-1j * 2.0 * np.pi * freqs_subarray * tau_sec)
            proj_err = np.linalg.norm((noise_subspace.conj().T @ a))**2
            spectrum[i] = 1.0 / (proj_err + 1e-15)
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        # pick peaks from spectrum to supplement if needed
        estimated = []
        if taus_samples.size > 0:
            estimated = list(np.round(taus_samples, 3))
        # find peaks on pseudospectrum
        peaks, props = signal.find_peaks(spectrum, height=0.01, distance=2, prominence=0.01)
        if len(peaks) > 0:
            peak_vals = tau_grid[peaks]
            # append peaks not close to existing estimates
            for pv in peak_vals:
                if len(estimated) >= num_paths:
                    break
                if all(abs(pv - e) > 0.5 for e in estimated):
                    estimated.append(float(round(pv, 3)))

        # finalize estimated list (limit to num_paths)
        estimated = sorted(estimated)[:num_paths]

        return tau_grid, spectrum, estimated

    def matrix_pencil_tde(self, rx_signal, num_paths, search_range_samples=(0,50), pencil_L=None, resample_tol=1e-6, tau_grid_step=0.01):
        return self.data_processor.matrix_pencil_tde(rx_signal, num_paths, search_range_samples, pencil_L, resample_tol, tau_grid_step)
    
    def matrix_pencil_multiframe(self, rx_signal, num_paths, num_ltfs=4, search_range_samples=(0,50), pencil_L=None, resample_tol=1e-6, tau_grid_step=0.01):
        return self.data_processor.matrix_pencil_multiframe(rx_signal, num_paths, num_ltfs, search_range_samples, pencil_L, resample_tol, tau_grid_step)
    
    def music_tde_multiframe(self, rx_signal, num_paths, num_ltfs=4, search_range_samples=(0, 50), subarray_size=None, non_overlapping=False):
        return self.data_processor.music_tde_multiframe(rx_signal, num_paths, num_ltfs, search_range_samples, subarray_size, non_overlapping)
    
    def plot_results(self, true_delays, methods_data, channel_impulse=None):
        self.plotter.plot_results(true_delays, methods_data, channel_impulse, self.channel_width_mhz)
