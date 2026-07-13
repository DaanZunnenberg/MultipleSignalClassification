import numpy as np
from scipy import signal
from scipy.linalg import eigh
from typing import List, Tuple, Union, Optional

class DataProcessor:
    """
    Handles TDE algorithms and processing.
    """
    def __init__(self, data_gen):
        self.data_gen = data_gen
    
    def cross_correlation_tde(self, rx_signal: np.ndarray,
                             search_range_samples: Tuple[int, int] = (0, 50),
                             threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Conventional cross-correlation based Time Delay Estimation
        
        Parameters:
        rx_signal: Received signal
        search_range_samples: Delay search range in samples
        threshold: Peak detection threshold
        
        Returns:
        lags_segment: Delay search grid
        corr_segment: Cross-correlation values  
        estimated_delays: Estimated delays in samples
        """
        lltf_ref = self.data_gen.generate_lltf_time_domain()

        # Compute full linear cross-correlation and magnitude
        # Matched filter: convolve received signal with time-reversed conjugate of reference
        # This explicitly implements r_xy[k] = sum_n x[n] * conj(y[n+k]) for complex baseband.
        correlation = signal.fftconvolve(rx_signal, np.conj(lltf_ref[::-1]), mode='full')
        correlation = np.abs(correlation)

        # Normalize to peak 1
        if np.max(correlation) > 0:
            correlation = correlation / np.max(correlation)

        # Lag vector (index alignment with full correlation)
        lags = np.arange(len(correlation)) - (len(lltf_ref) - 1)

        # Restrict to requested search window
        valid_indices = np.where((lags >= search_range_samples[0]) & (lags <= search_range_samples[1]))[0]
        if len(valid_indices) == 0:
            return np.array([]), np.array([]), []

        lags_segment = lags[valid_indices]
        corr_segment = correlation[valid_indices]

        # Peak detection on the correlation segment
        peaks, _ = signal.find_peaks(corr_segment, height=threshold, distance=2)
        estimated_delays = lags_segment[peaks]

        return lags_segment, corr_segment, list(estimated_delays)

    def _compute_covariance_matrix(self, H_estimated: np.ndarray, 
                                 subarray_size: int,
                                 freqs_used: Optional[np.ndarray] = None,
                                 non_overlapping: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariance matrix with frequency smoothing
        
        Parameters:
        H_estimated: Channel frequency response (used subcarriers)
        subarray_size: Size of frequency subarrays
        freqs_used: Frequencies (Hz) corresponding to H_estimated entries
        non_overlapping: If True, form non-overlapping contiguous segments of length subarray_size
                         and use their outer products as snapshots instead of sliding windows.
        
        Returns:
        R: Covariance matrix
        freqs_subarray: Frequency values for subarray (for steering vectors)
        """
        N = len(H_estimated)

        if subarray_size >= N:
            # No smoothing - use all frequencies as a single snapshot
            X = H_estimated.reshape(-1, 1)
            R = X @ X.conj().T
            freqs_subarray = freqs_used.copy() if freqs_used is not None else np.arange(N)
            return R, freqs_subarray

        # Build snapshots depending on mode
        all_snapshots = []
        if non_overlapping:
            num_segments = N // subarray_size
            for s in range(num_segments):
                seg = H_estimated[s*subarray_size:(s+1)*subarray_size]
                if len(seg) == subarray_size:
                    all_snapshots.append(seg)
                    all_snapshots.append(np.conj(seg[::-1]))
        else:
            num_subarrays = N - subarray_size + 1
            for i in range(num_subarrays):
                all_snapshots.append(H_estimated[i:i + subarray_size])
            H_reversed = np.conj(H_estimated[::-1])
            for i in range(num_subarrays):
                all_snapshots.append(H_reversed[i:i + subarray_size])

        if len(all_snapshots) == 0:
            raise ValueError("No snapshots could be formed for the given subarray_size")

        # Form data matrix and covariance
        X = np.array(all_snapshots).T  # shape: (subarray_size, num_snapshots)
        R = (X @ X.conj().T) / X.shape[1]

        # Choose a representative frequency vector for the steering vector
        if freqs_used is not None:
            if not non_overlapping:
                # choose the central overlapping subarray
                start_idx = max(0, (N - subarray_size) // 2)
                freqs_subarray = freqs_used[start_idx:start_idx + subarray_size].copy()
                if freqs_subarray.shape[0] < subarray_size:
                    freqs_subarray = freqs_used[:subarray_size].copy()
            else:
                # choose the middle non-overlapping segment
                num_segments = max(1, N // subarray_size)
                seg_idx = num_segments // 2
                start_idx = seg_idx * subarray_size
                start_idx = min(max(0, start_idx), max(0, N - subarray_size))
                freqs_subarray = freqs_used[start_idx:start_idx + subarray_size].copy()
                if freqs_subarray.shape[0] < subarray_size:
                    freqs_subarray = freqs_used[:subarray_size].copy()
        else:
            freqs_subarray = np.arange(subarray_size)

        return R, freqs_subarray

    def _select_top_unique_peaks(self, spectrum: np.ndarray, tau_grid: np.ndarray,
                                 num_paths: int, min_distance: int = 5,
                                 rel_height: float = 0.05, prominence: float = 0.01) -> list:
        """Select up to num_paths peaks from spectrum using relative height and minimum distance.
        Returns delays (values on tau_grid) sorted ascending.
        """
        if len(spectrum) == 0:
            return []
        maxv = np.max(spectrum)
        height = rel_height * maxv if maxv > 0 else 0.0
        peaks, props = signal.find_peaks(spectrum, height=height, distance=min_distance, prominence=prominence)
        if len(peaks) == 0:
            return []
        peak_heights = spectrum[peaks]
        order = np.argsort(peak_heights)[::-1]
        selected = peaks[order[:min(num_paths, len(peaks))]]
        return sorted(list(tau_grid[selected]))

    def music_tde(self, rx_signal: np.ndarray, 
                  num_paths: int,
                  search_range_samples: Tuple[int, int] = (0, 50),
                  subarray_size: int = None,
                  non_overlapping: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        MUSIC Time Delay Estimation with frequency smoothing
        
        Parameters:
        rx_signal: Received signal
        num_paths: Number of multipath components
        search_range_samples: Delay search range in samples
        subarray_size: Size of frequency subarrays for smoothing
        non_overlapping: If True, use non-overlapping contiguous segments for covariance matrix
        
        Returns:
        tau_grid: Delay search grid
        music_spectrum: MUSIC pseudospectrum
        estimated_delays: Estimated delays in samples
        """
        # Extract OFDM symbol and remove cyclic prefix
        rx_symbol = rx_signal[self.data_gen.cp_len:self.data_gen.cp_len + self.data_gen.fft_size]
        
        # Frequency domain processing
        rx_freq = np.fft.fft(rx_symbol) / np.sqrt(self.data_gen.fft_size)
        
        # Use fftshift for proper frequency ordering
        rx_freq_shifted = np.fft.fftshift(rx_freq)
        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)
        
        # Use only non-zero subcarriers
        used_mask = ltf_freq_shifted != 0
        H_estimated = rx_freq_shifted[used_mask] / ltf_freq_shifted[used_mask]
        
        # Get actual frequency values for used subcarriers
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.data_gen.fft_size, self.data_gen.ts))
        freqs_used = freqs_shifted[used_mask]
        
        # Set default subarray size if not provided
        if subarray_size is None:
            subarray_size = len(H_estimated) // 2
        
        # Compute covariance matrix with smoothing (optionally non-overlapping)
        R, freqs_subarray = self._compute_covariance_matrix(H_estimated, subarray_size, freqs_used, non_overlapping=non_overlapping)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Print eigenvalue information for debugging
        print(f"Eigenvalues: {eigenvalues[:min(10, len(eigenvalues))]}")
        
        # Noise subspace selection
        if len(eigenvalues) > num_paths:
            noise_subspace = eigenvectors[:, num_paths:]
        else:
            # Use all but the first eigenvector as noise subspace
            noise_subspace = eigenvectors[:, 1:]
        
        print(f"Noise subspace dimensions: {noise_subspace.shape}")
        print(f"Subarray size: {subarray_size}")
        
        # MUSIC spectrum computation
        start_sample, end_sample = search_range_samples
        tau_grid = np.arange(start_sample, end_sample, 0.01)
        music_spectrum = np.zeros(len(tau_grid))
        
        for i, tau_samples in enumerate(tau_grid):
            tau = tau_samples * self.data_gen.ts
            
            # Steering vector uses actual frequency values (Hz) for subarray rows
            a_tau = np.exp(-1j * 2 * np.pi * freqs_subarray * tau)
            
            # Project onto noise subspace - dimensions now match
            denominator = np.linalg.norm(noise_subspace.conj().T @ a_tau)**2
            music_spectrum[i] = 1 / (denominator + 1e-15)
        
        # Normalize spectrum
        if np.max(music_spectrum) > 0:
            music_spectrum = music_spectrum / np.max(music_spectrum)
        
        # Peak selection (use robust helper)
        estimated_delays = self._select_top_unique_peaks(music_spectrum, tau_grid, num_paths,
                                                         min_distance=5, rel_height=0.01, prominence=0.1)
            
        return tau_grid, music_spectrum, estimated_delays

    def matrix_pencil_tde(self,
                          rx_signal: np.ndarray,
                          num_paths: int,
                          search_range_samples: Tuple[int,int] = (0,50),
                          pencil_L: Optional[int] = None,
                          resample_tol: float = 1e-6,
                          tau_grid_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Matrix Pencil Time Delay Estimation (frequency-domain implementation).
        Returns (tau_grid, pseudospectrum, estimated_delays in samples).
        """
        # 1) Extract CFR like in music_tde
        rx_symbol = rx_signal[self.data_gen.cp_len:self.data_gen.cp_len + self.data_gen.fft_size]
        rx_freq = np.fft.fft(rx_symbol) / np.sqrt(self.data_gen.fft_size)
        rx_freq_shifted = np.fft.fftshift(rx_freq)
        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)

        used_mask = ltf_freq_shifted != 0
        H_est = rx_freq_shifted[used_mask] / ltf_freq_shifted[used_mask]
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.data_gen.fft_size, self.data_gen.ts))
        freqs_used = freqs_shifted[used_mask]

        # 2) Ensure (approximately) uniform spacing or resample onto a uniform grid
        if len(freqs_used) < 2:
            raise ValueError("Not enough subcarriers to run Matrix Pencil")

        df_vec = np.diff(freqs_used)
        if not np.allclose(df_vec, df_vec[0], rtol=resample_tol, atol=resample_tol):
            M = len(freqs_used)
            freqs = np.linspace(freqs_used[0], freqs_used[-1], M)
            H_real = np.interp(freqs, freqs_used, H_est.real)
            H_imag = np.interp(freqs, freqs_used, H_est.imag)
            H = H_real + 1j * H_imag
        else:
            H = H_est.copy()
            freqs = freqs_used.copy()

        M = len(H)
        if M <= num_paths:
            raise ValueError("Not enough frequency samples for Matrix Pencil with requested num_paths")

        # 3) Choose pencil length L and compute K (number of columns)
        if pencil_L is None:
            L = max(num_paths + 1, M // 2)
        else:
            L = int(pencil_L)
        # constrain L to valid range
        L = max(num_paths + 1, min(L, M - num_paths - 1))
        K = M - L
        if K <= 0:
            raise ValueError("Invalid pencil dimensions (L too large)")

        # 4) Build Hankel matrices Y0 and Y1 with K = M - L columns
        Y0 = np.zeros((L, K), dtype=complex)
        Y1 = np.zeros((L, K), dtype=complex)
        for k in range(K):
            Y0[:, k] = H[k:k + L]
            Y1[:, k] = H[k + 1:k + L + 1]

        # 5) SVD on Y0 and rank truncation
        U, S, Vh = np.linalg.svd(Y0, full_matrices=False)
        r = num_paths
        if r >= U.shape[1]:
            raise ValueError("num_paths too large for chosen pencil length")
        U_r = U[:, :r]
        S_r = S[:r].copy()
        V_r = Vh.conj().T[:, :r]

        # numerical safeguard for singular values
        eps = 1e-12
        S_r_safe = np.where(S_r < eps, eps, S_r)
        Sigma_inv = np.diag(1.0 / S_r_safe)

        S_mat = U_r.conj().T @ Y1 @ V_r @ Sigma_inv

        # 6) Eigenvalues -> poles z
        eigvals = np.linalg.eigvals(S_mat)
        z = eigvals

        # 7) Map poles to delays (handle wrapping/ambiguity)
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        H = H[sort_idx]
        df_val = np.mean(np.diff(freqs)) if len(freqs) > 1 else 0.0

        if df_val == 0.0:
            taus_sec_raw = np.zeros_like(z, dtype=float)
        else:
            taus_sec_raw = -np.angle(z) / (2.0 * np.pi * df_val)

        period = 1.0 / df_val if df_val != 0 else np.inf
        start_s = search_range_samples[0] * self.data_gen.ts
        end_s = search_range_samples[1] * self.data_gen.ts

        taus_sec = np.zeros_like(taus_sec_raw)
        for ii, t in enumerate(taus_sec_raw):
            if np.isfinite(period) and period > 0:
                center = 0.5 * (start_s + end_s)
                n = np.round((center - t) / period)
                taus_sec[ii] = t + n * period
            else:
                taus_sec[ii] = t

        taus_samples = np.real(taus_sec / self.data_gen.ts)

        # 8) Estimate amplitudes via LS using actual frequencies
        A = np.exp(-1j * 2.0 * np.pi * np.outer(freqs, taus_sec))  # M x r
        try:
            alpha, *_ = np.linalg.lstsq(A, H, rcond=None)
        except Exception:
            alpha = np.zeros((r,), dtype=complex)

        # 9) Build signal subspace from Vandermonde (QR)
        Q, _ = np.linalg.qr(A)
        U_s = Q[:, :r]

        # 10) Pseudospectrum on tau grid
        start_sample, end_sample = search_range_samples
        tau_grid = np.arange(start_sample, end_sample, tau_grid_step)
        spectrum = np.zeros(len(tau_grid))
        I = np.eye(len(freqs), dtype=complex)
        for i, tau_samp in enumerate(tau_grid):
            tau_sec = tau_samp * self.data_gen.ts
            a = np.exp(-1j * 2.0 * np.pi * freqs * tau_sec)
            proj_err = np.linalg.norm((I - U_s @ U_s.conj().T) @ a)**2
            spectrum[i] = 1.0 / (proj_err + 1e-15)

        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        # 11) Peak selection (robust)
        estimated = self._select_top_unique_peaks(spectrum, tau_grid, num_paths, min_distance=2, rel_height=0.01, prominence=0.01)

        # debug
        print(f"Matrix Pencil: L={L}, K={K}, M={M}, estimated={estimated}")
        return tau_grid, spectrum, estimated

    def matrix_pencil_multiframe(self,
                                 rx_signal: np.ndarray,
                                 num_paths: int,
                                 num_ltfs: int = 4,
                                 search_range_samples: Tuple[int,int] = (0,50),
                                 pencil_L: Optional[int] = None,
                                 resample_tol: float = 1e-6,
                                 tau_grid_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Multi-frame Matrix Pencil: average pseudospectra from multiple L-LTF symbols.
        This implements a simple ensemble average of per-frame MP pseudospectra.
        """
        lltf_len = self.data_gen.fft_size + self.data_gen.cp_len
        sum_spec = None
        count = 0
        tau_grid = None

        for i in range(num_ltfs):
            start = i * lltf_len + self.data_gen.cp_len
            end = start + self.data_gen.fft_size
            if end > len(rx_signal):
                break

            rx_sym = rx_signal[start:end]
            # build temporary rx buffer with cp_len at front so matrix_pencil_tde extracts symbol
            tmp_rx = np.concatenate([np.zeros(self.data_gen.cp_len, dtype=complex), rx_sym])
            t_grid, spec, est = self.matrix_pencil_tde(tmp_rx, num_paths, search_range_samples,
                                                      pencil_L=pencil_L, resample_tol=resample_tol,
                                                      tau_grid_step=tau_grid_step)
            if tau_grid is None:
                tau_grid = t_grid
            else:
                # ensure same grid
                if len(tau_grid) != len(t_grid) or not np.allclose(tau_grid, t_grid):
                    # interpolate spec onto tau_grid
                    spec = np.interp(tau_grid, t_grid, spec)

            if sum_spec is None:
                sum_spec = spec.copy()
            else:
                sum_spec += spec
            count += 1

        if count == 0:
            raise ValueError("No L-LTF symbols found for multi-frame MP")

        avg_spec = sum_spec / count
        # Peak selection on averaged spectrum (robust)
        estimated = self._select_top_unique_peaks(avg_spec, tau_grid, num_paths, min_distance=2, rel_height=0.01, prominence=0.01)

        print(f"Matrix Pencil multiframe: processed {count} frames, returning {len(avg_spec)}-point avg spectrum")
        # Debug print
        print(f"Multiframe MP tau_grid len: {len(tau_grid)} avg_spec len: {len(avg_spec)} estimated: {estimated}")
        return tau_grid, avg_spec, estimated

    def music_tde_multiframe(self, rx_signal: np.ndarray,
                             num_paths: int,
                             num_ltfs: int = 4,
                             search_range_samples: Tuple[int, int] = (0, 50),
                             subarray_size: int = None,
                             non_overlapping: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        MUSIC TDE with multiple L-LTF observations
        
        Parameters:
        rx_signal: Received signal containing multiple L-LTFs
        num_paths: Number of multipath components
        num_ltfs: Number of L-LTFs to use
        search_range_samples: Delay search range in samples
        subarray_size: Size of frequency subarrays for smoothing
        non_overlapping: If True, use non-overlapping contiguous segments for covariance matrix
        
        Returns:
        tau_grid: Delay search grid
        music_spectrum: MUSIC pseudospectrum
        estimated_delays: Estimated delays in samples
        """
        lltf_len = self.data_gen.fft_size + self.data_gen.cp_len
        R_averaged = None
        count = 0
        
        # Frequency setup
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.data_gen.fft_size, self.data_gen.ts))
        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)
        used_mask = ltf_freq_shifted != 0
        freqs_used = freqs_shifted[used_mask]
        
        # Set default subarray size if not provided
        if subarray_size is None:
            subarray_size = len(freqs_used) // 2
        
        # Process each L-LTF
        for i in range(num_ltfs):
            start = i * lltf_len + self.data_gen.cp_len
            end = start + self.data_gen.fft_size
            
            if end > len(rx_signal):
                break
                
            # Extract and process L-LTF symbol
            rx_sym = rx_signal[start:end]
            rx_f = np.fft.fftshift(np.fft.fft(rx_sym) / np.sqrt(self.data_gen.fft_size))
            H_est = rx_f[used_mask] / ltf_freq_shifted[used_mask]
            
            # Compute covariance for this L-LTF (optionally non-overlapping segments)
            R_curr, freqs_subarray = self._compute_covariance_matrix(H_est, subarray_size, freqs_used, non_overlapping=non_overlapping)
            
            # Average covariance matrices (safe accumulation)
            if R_averaged is None:
                # make a copy to avoid accidental modification of R_curr elsewhere
                R_averaged = R_curr.copy()
                count = 1
            else:
                R_averaged += R_curr
                count += 1
            
        if R_averaged is None:
            raise ValueError("No valid L-LTFs processed")
            
        R = R_averaged / count
        
        # Eigen decomposition
        eigenvalues, eigenvectors = eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace selection
        if len(eigenvalues) > num_paths:
            noise_subspace = eigenvectors[:, num_paths:]
        else:
            noise_subspace = eigenvectors[:, 1:]
        
        # MUSIC spectrum computation
        start_sample, end_sample = search_range_samples
        tau_grid = np.arange(start_sample, end_sample, 0.1)
        music_spectrum = np.zeros(len(tau_grid))
        
        for i, tau_samples in enumerate(tau_grid):
            tau = tau_samples * self.data_gen.ts
            # Use subarray frequency values (Hz) for steering vector
            a_tau = np.exp(-1j * 2 * np.pi * freqs_subarray * tau)
            
            denominator = np.linalg.norm(noise_subspace.conj().T @ a_tau)**2
            music_spectrum[i] = 1 / (denominator + 1e-15)
        
        # Normalize spectrum
        if np.max(music_spectrum) > 0:
            music_spectrum = music_spectrum / np.max(music_spectrum)
        
        # Peak detection
        peaks, properties = signal.find_peaks(music_spectrum, 
                                            height=0.2, 
                                            distance=5,
                                            prominence=0.1)
        estimated_delays = []
        
        if len(peaks) > 0:
            peak_heights = music_spectrum[peaks]
            top_peaks = peaks[np.argsort(peak_heights)[::-1][:min(num_paths, len(peaks))]]
            estimated_delays = sorted(list(tau_grid[top_peaks]))
            
        return tau_grid, music_spectrum, estimated_delays
