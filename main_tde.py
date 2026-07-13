import numpy as np
from music_tde import MUSICTDE
from typing import Tuple, Optional

def demonstrate_music_tde():
    """
    Demonstration of MUSIC Time Delay Estimation for WiFi L-LTF
    """
    # Initialize with 20MHz channel
    music_tde = MUSICTDE(
        channel_width_mhz=20,
        snr_db=0.0,
        realistic_lltf=False,
        freq_offset_hz=1e2,               # 0.1 kHz carrier offset
        phase_noise_std=1e-3,             # small per-sample phase-noise increment (rad)
        timing_jitter_std=0.1,            # fractional sample timing jitter (std)
        iq_gain_imbalance=0.02,           # 2% I/Q gain imbalance
        iq_phase_imbalance_deg=1.0        # 1 degree I/Q phase mismatch
    )
    
    # Generate transmitted signal
    lltf_tx = music_tde.generate_lltf_time_domain()
    
    # Define multipath channel with fractional delays
    delays_samples = [2.5, 11.5] 
    # amplitudes = [1.0, 1 * np.exp(1j * 0.5), 1 * np.exp(1j * 1.2)]
    amplitudes = [1, 1]
    
    print("MUSIC Time Delay Estimation Demo")
    print("=" * 50)
    print(f"Channel Bandwidth: {music_tde.channel_width_mhz}MHz")
    print(f"True Delays: {delays_samples} samples")
    print(f"FFT Size: {music_tde.fft_size}, CP Length: {music_tde.cp_len}")
    print(f"SNR: {music_tde.snr_db} dB")

    # Create channel and transmit
    channel = music_tde.create_multipath_channel(delays_samples, amplitudes)
    rx_signal = music_tde.transmit_through_channel(lltf_tx, channel)
    
    # Determine number of paths from ground truth
    num_paths = len(delays_samples)
    # Method 1: Cross-correlation (baseline)
    lags, corr, est_cc = music_tde.cross_correlation_tde(rx_signal, (0, 35))
    print(f"\n1. Cross-Correlation Estimates: {est_cc}")
    
    # Method 2: MUSIC with frequency smoothing
    subarray_size = 44
    print("\nComputing MUSIC TDE...")
    tau, spec, est_music = music_tde.music_tde(rx_signal, num_paths, (0, 35), subarray_size, non_overlapping=False)
    # Non-overlapping segments variant
    tau_no, spec_no, est_music_no = music_tde.music_tde(rx_signal, num_paths, (0, 35), subarray_size, non_overlapping=True)
    # Method 2b: ESPRIT
    print("\nComputing ESPRIT TDE...")
    tau_es, spec_es, est_es = music_tde.esprit_tde(rx_signal, num_paths, (0,35), subarray_size=subarray_size, non_overlapping=False, tau_grid_step=0.01)
    print(f"ESPRIT Estimates: {est_es}")
    # Method X: Matrix Pencil TDE (single-symbol)
    print("\nComputing Matrix Pencil TDE...")
    tau_mp, spec_mp, est_mp = music_tde.matrix_pencil_tde(rx_signal, num_paths, (0, 35), pencil_L=None, tau_grid_step=0.01)
    print(f"2. MUSIC Estimates: {est_music}")
    
    # Method 3: Multi-frame MUSIC
    print("\nComputing Multi-frame MUSIC TDE...")
    lltf_frame = np.tile(lltf_tx, 4)
    rx_frame = music_tde.transmit_through_channel(lltf_frame, channel)
    tau_mf, spec_mf, est_mf = music_tde.music_tde_multiframe(rx_frame, num_paths, 4, (0, 35), subarray_size, non_overlapping=False)
    # Non-overlapping multiframe variant
    tau_mf_no, spec_mf_no, est_mf_no = music_tde.music_tde_multiframe(rx_frame, num_paths, 4, (0, 35), subarray_size, non_overlapping=True)
    print(f"3. Multi-frame MUSIC Estimates: {est_mf}")

    # Multi-frame Matrix Pencil
    print("\nComputing Multi-frame Matrix Pencil TDE...")
    tau_mp_mf, spec_mp_mf, est_mp_mf = music_tde.matrix_pencil_multiframe(rx_frame, num_paths, num_ltfs=4, search_range_samples=(0,35), pencil_L=None, tau_grid_step=0.01)
    print(f"Multi-frame MP Estimates: {est_mp_mf}")

    # Plot results
    methods_data = {
        'Cross-Correlation': (lags, corr, est_cc),
        'MUSIC': (tau, spec, est_music),
        'ESPRIT': (tau_es, spec_es, est_es),
        'Multi-frame MUSIC': (tau_mf, spec_mf, est_mf),
        'Matrix Pencil': (tau_mp, spec_mp, est_mp),
        'Matrix Pencil MF': (tau_mp_mf, spec_mp_mf, est_mp_mf),
        # overlays (non-overlapping segment variants) plotted on top of MUSIC axes
        'MUSIC_nonoverlap': (tau_no, spec_no, est_music_no),
        'Multi-frame MUSIC_nonoverlap': (tau_mf_no, spec_mf_no, est_mf_no)
    }
    
    music_tde.plot_results(delays_samples, methods_data, channel)


def esprit_tde(self, rx_signal: np.ndarray,
                   num_paths: int,
                   search_range_samples: Tuple[int,int] = (0,50),
                   subarray_size: int = None,
                   non_overlapping: bool = False,
                   tau_grid_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        ESPRIT-based TDE using frequency smoothing (shift-invariance on subarrays).
        Returns (tau_grid, pseudospectrum, estimated_delays)
        """
        # Extract CFR like in music_tde
        rx_symbol = rx_signal[self.data_gen.cp_len:self.data_gen.cp_len + self.data_gen.fft_size]
        rx_freq = np.fft.fft(rx_symbol) / np.sqrt(self.data_gen.fft_size)
        rx_freq_shifted = np.fft.fftshift(rx_freq)
        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)

        used_mask = ltf_freq_shifted != 0
        H_estimated = rx_freq_shifted[used_mask] / ltf_freq_shifted[used_mask]
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.data_gen.fft_size, self.data_gen.ts))
        freqs_used = freqs_shifted[used_mask]

        # default subarray size
        if subarray_size is None:
            subarray_size = len(H_estimated) // 2

        # build covariance and get representative freqs
        R, freqs_subarray = self._compute_covariance_matrix(H_estimated, subarray_size, freqs_used, non_overlapping=non_overlapping)

        # EVD and signal subspace
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        # signal subspace (L x P)
        if num_paths <= eigvecs.shape[1]:
            U_s = eigvecs[:, :num_paths]
        else:
            U_s = eigvecs

        # Partition U_s into two overlapping subarrays (shift by one row)
        if U_s.shape[0] < 2:
            return np.array([]), np.array([]), []
        U1 = U_s[:-1, :]
        U2 = U_s[1:, :]

        # Solve least-squares mapping U1*Phi = U2 -> Phi = pinv(U1) @ U2
        try:
            Phi = np.linalg.pinv(U1) @ U2
            eigvals_phi, _ = np.linalg.eig(Phi)
        except Exception:
            eigvals_phi = np.array([])

        # Map eigenvalues to delays
        if len(freqs_subarray) > 1 and len(eigvals_phi) > 0:
            df = np.mean(np.diff(freqs_subarray))
            taus_sec = -np.angle(eigvals_phi) / (2.0 * np.pi * df)
            taus_samples = np.real(taus_sec / self.data_gen.ts)
        else:
            taus_samples = np.array([])

        # Build pseudospectrum using noise subspace (for plotting)
        eigenvalues_full, eigenvectors_full = np.linalg.eigh(R)
        idx_full = np.argsort(eigenvalues_full)[::-1]
        eigenvectors_full = eigenvectors_full[:, idx_full]
        if eigenvectors_full.shape[1] > num_paths:
            noise_subspace = eigenvectors_full[:, num_paths:]
        else:
            noise_subspace = eigenvectors_full[:, 1:]

        start_sample, end_sample = search_range_samples
        tau_grid = np.arange(start_sample, end_sample, tau_grid_step)
        spectrum = np.zeros(len(tau_grid))
        for i, tau_samp in enumerate(tau_grid):
            tau_sec = tau_samp * self.data_gen.ts
            a = np.exp(-1j * 2.0 * np.pi * freqs_subarray * tau_sec)
            proj_err = np.linalg.norm((noise_subspace.conj().T @ a))**2
            spectrum[i] = 1.0 / (proj_err + 1e-15)
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        # Select top peaks on spectrum as final ESPRIT-friendly estimates
        estimated = self._select_top_unique_peaks(spectrum, tau_grid, num_paths, min_distance=2, rel_height=0.01, prominence=0.01)
        # If ESPRIT produced direct taus, merge/prioritize them (map into search range)
        if taus_samples.size > 0:
            valid = [t for t in taus_samples if start_sample - 1 <= t <= end_sample + 1]
            if len(valid) > 0:
                combined = sorted(set([float(round(v, 3)) for v in valid] + [float(round(v, 3)) for v in estimated]))
                estimated = combined[:num_paths]

        return tau_grid, spectrum, estimated


if __name__ == "__main__":
    demonstrate_music_tde()