import numpy as np
import scipy, data_generator, data_processor, inspect
import sys, os, glob
from typing import Any, Callable, List, Tuple, Union, Optional
from tqdm import tqdm
from scipy import signal
from scipy.linalg import eigh


from music_tde import *
from ctypes import *

import warnings

def deprecated(message):
  def deprecated_decorator(func):
      def deprecated_func(*args, **kwargs):
          warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                        category=DeprecationWarning,
                        stacklevel=2)
          warnings.simplefilter('default', DeprecationWarning)
          return func(*args, **kwargs)
      return deprecated_func
  return deprecated_decorator

class DataProcessorV2:
    """
    Handles TDE algorithms and processing. Updated version

    The class imports all the attributes from the data_processor.DataProcessor and only makes changes to the music_tde function
    """
    def __init__(self, data_gen: Any = False, warm_update: bool = True, **kwargs):
        if not data_gen: 
            print(f'Module .data_generator.DataGenerator called at {id(self)}')
            self.data_gen = self._data_gen()
        else: self.data_gen = data_gen
        self.DataProcessor: data_processor.DataProcessor = data_processor.DataProcessor(data_gen = data_gen)
        if self.channel_width_mhz == 20:
            self.fft_size = 64
            self.cp_len = 16
            self.sample_rate = 20e6
            self.active_subcarriers = 52
        elif self.channel_width_mhz == 40:
            self.fft_size = 128
            self.cp_len = 32
            self.sample_rate = 40e6
            self.active_subcarriers = 114
        else:
            raise ValueError("Channel width must be 20 or 40 MHz")
        
        self.ts = 1.0 / self.sample_rate
        self.subcarrier_spacing = self.sample_rate / self.fft_size
        if warm_update: print('DataProcessorV2 updated version')

    def _data_gen(self, **kwargs) -> Any:
        if len(kwargs) == 0: 
            kwargs = {
                        "channel_width_mhz"         :   20, 
                        "snr_db"                    :   0.0,
                        "realistic_lltf"            :   False,
                        "freq_offset_hz"            :   1e2,
                        "phase_noise_std"           :   1e-3,
                        "timing_jitter_std"         :   0.1,
                        "iq_gain_imbalance"         :   0.02,
                        "iq_phase_imbalance_deg"    :   1.0
                    }
        
            # Store the parameters in kwargs as self attributes and store the data_gen object if none is provided in __init__
            for _, __ in kwargs.items(): self.__setattr__(_, __)
            return data_generator.DataGenerator(**kwargs)
        else: pass

class MUSICTDEV2(DataProcessorV2):
    """
    MUSIC-based Time Delay Estimation for IEEE 802.11n WiFi L-LTF
    Implements superresolution delay estimation using subspace methods
    with frequency smoothing for enhanced performance.
    """

    def __init__(self, data_gen = False, warm_update = True, **kwargs):
        super().__init__(data_gen, warm_update, **kwargs)
        for _, __ in kwargs.items(): self.__setattr__(_, __)
        if not kwargs.get('num_path', False): 
            self.num_path = 1
            self.N = self.num_path

    @staticmethod
    def _snapshot(N: int, L: int, H: np.matrix[complex], rX: bool = False) -> Any:
        try:
            H_conj: np.matrix[complex] = np.conj(H[::-1])
            
            X: np.array[complex] = np.array(
                [H[i:i+L] for i in range(N - L + 1)] + [H_conj[i:i+L] for i in range(N - L + 1)]
            ).T
            R: Any = (X @ X.conj().T) / X.shape[1]
        except:
            X = H.reshape(-1, 1)
            R = X @ X.conj().T

        if not rX: return R
        else: return R, X

    @staticmethod
    def _snapshot_simple(N: int, L: int, H: np.matrix[complex], rX: bool = False, O: bool = True) -> Any:
        all_snapshots = []
        if not O:
            num_segments = N // L
            for s in range(num_segments):
                seg = H[s*L:(s+1)*L]
                if len(seg) == L:
                    all_snapshots.append(seg)
                    all_snapshots.append(np.conj(seg[::-1]))
        else:
            num_subarrays = N - L + 1
            for i in range(num_subarrays):
                all_snapshots.append(H[i:i + L])
            H_reversed = np.conj(H[::-1])
            for i in range(num_subarrays):
                all_snapshots.append(H_reversed[i:i + L])

        try:
            X: np.array[complex] = np.array(all_snapshots).T
            R: Any = (X @ X.conj().T) / X.shape[1]
        except:
            X = H.reshape(-1, 1)
            R = X @ X.conj().T

        if not rX: return R
        else: return R, X
    
    @staticmethod
    def _SVD(N: int, R: np.matrix[complex], X: np.matrix[complex] = None) -> Any:
        """
        Singular value decomposition
        """
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        if N <= eigvecs.shape[1]:
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
        
        return eigvals_phi

    @staticmethod
    def _SVDV2(R: np.matrix[complex]) -> Any:
        """
        Singular value decomposition simple
        """
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        return eigvals[idx], eigvecs, idx
    
    def _matrix_pencil(self, 
                       rx_signal: np.matrix[complex],
                       fft: callable = np.fft.fft,
                       method: str = 'interval',
                       subarray_size: int = 44,
                       O: bool = True) -> Any:
        """
        The matrix pencil method to define the signal and noice subspaces
        """
        assert method == 'interval'




        ltf_freq_shifted = np.fft.fftshift(self.data_gen.l_ltf_freq)
        
        rx_freq = fft(
            rx_signal[self.cp_len:self.cp_len + self.fft_size]
        ) / np.sqrt(self.data_gen.fft_size)

        plt.plot(rx_signal)
        plt.show()
        rx_freq_shifted = np.fft.fftshift(rx_freq)

        used_mask = ltf_freq_shifted != 0
        H_estimated = rx_freq_shifted[used_mask] / ltf_freq_shifted[used_mask]
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(self.data_gen.fft_size, self.data_gen.ts))
        freqs_used = freqs_shifted[used_mask]

        N = len(H_estimated)
        L = subarray_size
        R, X = self._snapshot_simple(N = N, L = L, H = H_estimated, rX = True, O = O)

        EVal, EVec, IDX = self._SVDV2(R = R)
        return subarray_size, freqs_used, N, L, R, EVal, EVec, IDX

    def _private(self, _matrix_pencil: Callable = False, **kwargs) -> Any: 
        if _matrix_pencil: return _matrix_pencil(**kwargs)
        else: return self._matrix_pencil(**kwargs)

    @staticmethod
    def _frequency(_fu: Any = None, 
                   overlapping: bool = True,
                   array_size: int = 1,
                   sub_array_size: int = 1) -> Any:
        if _fu is None: return None
        else:
            try:
                assert sub_array_size <= array_size
                frequency_array_size: int = len(_fu)
                if not overlapping:
                    # Select the central overlapping sub array
                    start_index: int = max(0, (array_size - sub_array_size) // 2)
                else:
                    # Select the middle non-overlapping segment
                    segment_number: int = max(1, array_size // sub_array_size)
                    segment_index : int = segment_number // 2
                    start_index   : int = min(max(0,segment_index * sub_array_size), max(0, array_size - sub_array_size))
                
                if frequency_array_size < start_index + sub_array_size:
                    frequency_sub_array: Any = _fu[:sub_array_size].copy()
                else:
                    frequency_sub_array: Any = _fu[start_index : start_index + sub_array_size].copy()

                return frequency_sub_array
            
            except AssertionError as e:
                print(e)
                return None


    def _music_tde(self, 
                   rx_signal: np.matrix[complex], 
                   _matrix_pencil: Callable = False, 
                   method: str = 'interval',
                   **kwargs) -> Any:
        """
        MUSIC Time Delay Estimation with frequency smoothing
        
        Parameters:
        matrix_pencil: The SVD matrix method
        
        Returns:
        tau_grid: Delay search grid
        music_spectrum: MUSIC pseudospectrum
        estimated_delays: Estimated delays in samples
        """
        if not method == 'interval':
            raise NotImplementedError('Interval method supported. See music_tde.MUSICTDE for detail')
        else:
            if not _matrix_pencil: _matrix_pencil = self._matrix_pencil
            sub_array_size, freqs_used, array_size, L, R, EVal, EVec, IDX = self._private(
                rx_signal = rx_signal, _matrix_pencil = _matrix_pencil
            )
            print(f"Eigenvalues:", *EVal[:min(10, len(EVal))], sep = '\n')      
            overlapping: bool = kwargs.get('overlapping', True)
            freq_used: Any = self.__dict__.get('freqs_used', freqs_used)
            try:
                frequency_sub_array = self._frequency(
                    _fu             = freq_used,
                    overlapping     = overlapping,
                    array_size      = array_size,
                    sub_array_size  = sub_array_size,
                )
            except:
                frequency_sub_array = np.arange(sub_array_size)

            try: 
                assert len(EVal) > self.num_path
                noise_subspace = EVec[:, self.num_path:]
            except AssertionError as e: noise_subspace = EVec[:, 1:]

            print(f"Noise subspace dimensions: {noise_subspace.shape}")
            print(f"Subarray size: {sub_array_size}")

            # Finding the optimal tau
            start_sample, end_sample = (0,50)
            tau_grid = np.arange(start_sample, end_sample, 0.01)
            music_spectrum = np.zeros(len(tau_grid))
            
            for i, tau_samples in enumerate(tau_grid):
                tau = tau_samples * self.data_gen.ts
                
                # Steering vector uses actual frequency values (Hz) for subarray rows
                a_tau = np.exp(-1j * 2 * np.pi * frequency_sub_array * tau)
                
                # Project onto noise subspace - dimensions now match
                denominator = np.linalg.norm(noise_subspace.conj().T @ a_tau)**2
                music_spectrum[i] = 1 / (denominator)

            plt.plot(music_spectrum)


            tau: np.array[float] = np.array([])


            

    def run(self) -> ...:...
    def main(self) -> ...:...
   

def _compute_covariance_matrix(H_estimated: np.ndarray, 
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

class MUSICTDEV3(MUSICTDE):

    def __init__(self, 
                 channel_width_mhz = 20, 
                 snr_db = 40, 
                 realistic_lltf = True, 
                 freq_offset_hz = 0, 
                 phase_noise_std = 0, 
                 timing_jitter_std = 0, 
                 iq_gain_imbalance = 0, 
                 iq_phase_imbalance_deg = 0,
                 test_kwargs: bool = True):
        if not test_kwargs: super().__init__(channel_width_mhz, 
                         snr_db, 
                         realistic_lltf, 
                         freq_offset_hz, 
                         phase_noise_std, 
                         timing_jitter_std, 
                         iq_gain_imbalance, 
                         iq_phase_imbalance_deg)
        else: super().__init__(**self._test_kwargs)
        
    @property
    def _test_kwargs(self) -> dict:
        kwargs = {
                    "channel_width_mhz"         :   20, 
                    "snr_db"                    :   0.0,
                    "realistic_lltf"            :   False,
                    "freq_offset_hz"            :   1e2,
                    "phase_noise_std"           :   1e-3,
                    "timing_jitter_std"         :   0.1,
                    "iq_gain_imbalance"         :   0.02,
                    "iq_phase_imbalance_deg"    :   1.0
                }
        return kwargs    
    
    @classmethod
    def _DataProcessor(DataProcessor: DataProcessor = DataProcessor, **kwargs):...
    
    @staticmethod
    def _compute_covariance_matrix(*args, **kwargs) -> Any: return _compute_covariance_matrix(*args, **kwargs)

    def _preprocessor(self, delay: list = [2.5], ampl: list = [1], num_paths: int = 1) -> None:
        """
        Generates a rx_signal with the added delay and amplitude
        """
        self.num_paths : int = num_paths
        try:
            assert len(ampl) == len(delay)
            self.rx_signal = self.transmit_through_channel(
                self.generate_lltf_time_domain(), 
                self.create_multipath_channel(delay, ampl)
                )
        except AssertionError as e: self.rx_signal = None

        self.rx_symbol: Any = self.rx_signal[
            self.data_gen.cp_len : self.data_gen.cp_len + self.data_gen.fft_size
        ]

        self.rx_freq_shifted: Any = np.fft.fftshift(
            np.fft.fft(self.rx_symbol) / np.sqrt(self.data_gen.fft_size)
        )

        self.ltf_freq_shifted: Any = np.fft.fftshift(
            self.data_gen.l_ltf_freq
        )

        self.mask: bool = self.ltf_freq_shifted != 0
        self.H: np.matrix[complex] = self.rx_freq_shifted[self.mask] / self.ltf_freq_shifted[self.mask]

        self.freqs_shifted: Any = np.fft.fftshift(
            np.fft.fftfreq(
                self.data_gen.fft_size, 
                self.data_gen.ts
            )
        )
        self.freqs_used: Any = self.freqs_shifted[self.mask]
    
    def _eig_decom(self, 
                   sub_array_size: Any = None,
                   nonoverlapping: bool = False) -> Any:
        try:
            assert hasattr(self, 'H')
            self.sub_array_size: int = len(self.H) // 2 if sub_array_size is None else sub_array_size
        except AssertionError as e: 
            print('@Self.H not found')
            pass
        self.R, self.F = self._compute_covariance_matrix(
            self.H, 
            self.sub_array_size, 
            self.freqs_used, 
            nonoverlapping)
        
        eigenvalues, eigenvectors = eigh(self.R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"Eigenvalues: {eigenvalues[:min(10, len(eigenvalues))]}")
        if len(eigenvalues) > self.num_paths: self.noise_subspace = eigenvectors[:, self.num_paths:]
        else: self.noise_subspace = eigenvectors[:, 1:]
        
        print(f"Noise subspace dimensions: {self.noise_subspace.shape}")
        print(f"Subarray size: {self.sub_array_size}")
        return eigenvalues, eigenvectors, idx

    def _run(self,
        SRS     : Tuple[int, int]   = (0,50),
        err     : float             = 1e-1,
        ts      : float             = 5e-8,
        FSA     : np.array          = None, 
        TS      : float             = None,
        NSP     : np.matrix         = None,
        ) -> Any:
        try:
            assert FSA is not None
            assert TS  is not None
            assert NSP is not None

            start, end, *_ = SRS
            projections: np.array[float] = np.array([
                np.exp(-1j * 2 * np.pi * FSA * tau * TS) for tau in np.arange(start, end, err)
            ]).T
            return np.linalg.norm(NSP.conj().T @ projections, axis = 0)

        except Exception as e: pass

    
    def main(self, *args, **kwargs) -> Any:
        """
        Estimates the time delays.

        The estimation method may be improved by implementing a more clever search algorithm, computationally more attractive
        """
        raise NotImplementedError
        self.sub_array_size: int = kwargs.pop('sub_array_size', None)
        self._preprocessor()
        
        E: tuple[Any, Any, Any] = self._eig_decom(sub_array_size = self.sub_array_size)
        Estimates: list[float] = self._run(FSA = self.F, ts = self.data_gen.ts, TS = self.ts, NSP = self.noise_subspace, **kwargs)
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Exzmple
    music = MUSICTDEV3()
    music.main(sub_array_size = 44)
