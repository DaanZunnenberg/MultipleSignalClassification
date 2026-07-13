import numpy as np
from musicssvd import MUSICTDE

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


if __name__ == "__main__":
    demonstrate_music_tde()