# forward.py

import numpy as np
from music_tde import MUSICTDE
import matplotlib.pyplot as plt
#from typing import Tuple, Optional
# from time_varying import tv
from scipy import signal
#from matplotlib.animation import FuncAnimation 

print("")

single_delay = True
plot_individual = False
animate = False

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


def transmit(tx_signal, channel, freq_offset_hz=100.0, ts=5e-8, 
             phase_noise_std=0.001,
             iq_gain_imbalance=0.02,
             iq_phase_imbalance_deg=1.0,
             snr_db = 0.0,
             add_noise = True):
        rx_signal = signal.fftconvolve(tx_signal, channel, mode='full')

        # Apply carrier frequency offset (CFO) across the received signal if specified
        N = len(rx_signal)
        if freq_offset_hz != 0.0 or phase_noise_std > 0.0:
            n = np.arange(N)
            # frequency offset term
            if freq_offset_hz != 0.0:
                cfo = np.exp(1j * 2.0 * np.pi * freq_offset_hz * n * ts)
            else:
                cfo = np.ones(N, dtype=complex)
            # phase noise: cumulative Wiener-like phase noise (small increments)
            if phase_noise_std > 0.0:
                increments = np.random.randn(N) * phase_noise_std
                phase_noise = np.cumsum(increments)
                ph_noise = np.exp(1j * phase_noise)
                # print("Adding phase noise!")
            else:
                ph_noise = np.ones(N, dtype=complex)
            rx_signal = rx_signal * cfo * ph_noise

        # Apply IQ imbalance if requested (simple gain & phase imbalance model)
        if abs(iq_gain_imbalance) > 0.0 or abs(iq_phase_imbalance_deg) > 0.0:
            g = iq_gain_imbalance
            phi = np.deg2rad(iq_phase_imbalance_deg)
            # apply gain imbalance and phase rotation between I and Q
            i_part = np.real(rx_signal) * (1.0 + g)
            q_part = np.imag(rx_signal) * (1.0 - g)
            rx_signal = (i_part + 1j * q_part) * np.exp(1j * phi)

        # Add AWGN
        if add_noise:
            signal_power = np.mean(np.abs(rx_signal)**2)
            noise_power = signal_power / (10**(snr_db / 10)) if signal_power > 0 else 0.0
            noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        else:
            noise = 0.0
        return rx_signal + noise


# simpler version
def forward(tx_signal, channel):
    rx_signal = signal.fftconvolve(tx_signal, channel, mode='full')
    N = len(rx_signal)
    ts = 5e-8
    freq_offset_hz = 100.0
    if True:
        n = np.arange(N)
        cfo = np.exp(1j * 2.0 * np.pi * freq_offset_hz * n * ts)
        rx_signal = rx_signal * cfo
    return rx_signal 


lltf_tx = music_tde.generate_lltf_time_domain()

delays_samples = [2.5, 11.5] 
#amplitudes = [1.0, 1 * np.exp(1j * 0.5), 1 * np.exp(1j * 1.2)]
amplitudes = [1, 0.8]

channel = music_tde.create_multipath_channel(delays_samples, amplitudes)
rx_ideal = transmit(lltf_tx, channel, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)

rx_changed = transmit(lltf_tx, channel, add_noise = False)
rx_signal = transmit(lltf_tx, channel)
channel0 = music_tde.create_multipath_channel([2.5], [1])
rx_ideal0 = transmit(lltf_tx, channel0, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)
channel1 = music_tde.create_multipath_channel([5.0], [1])
rx_ideal1 = transmit(lltf_tx, channel1, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)

#
# define the grid
#
tau_grid = np.arange(0, 35, 0.01)
tau_grid = np.arange(0, 5, 0.01)
tau_grid = np.arange(0, 5, 0.05)
ntau = len(tau_grid)

#
# define the lengths
#
tem_len = [2, 4, 8, 16, 32, 64, 128]
ntem = len(tem_len)

dst = np.zeros((ntem, ntau))
dstn = np.zeros((ntem, ntau))

for k, template_len in enumerate(tem_len):

    # Shorten everything
    ifirst = 16
    ilast = ifirst + template_len
    lltf_tx1   = lltf_tx[ifirst:ilast]
    rx_signal1 = rx_signal[ifirst:ilast]
    rx_ideal1 = rx_ideal[ifirst:ilast]
    nshort = len(lltf_tx1)

    test_signals = np.zeros((ntau, nshort),dtype='complex')
    for i, tau in enumerate(tau_grid):
        channel1 = music_tde.create_multipath_channel([tau, 11.5], [1, 0.8])
        if single_delay:
            channel1 = music_tde.create_multipath_channel([tau], [1])
        rx1 = transmit(lltf_tx, channel1, phase_noise_std=0.0, 
                              iq_gain_imbalance=0.0, 
                              iq_phase_imbalance_deg=0.0,
                              add_noise = False)
    
        rx1 = rx1[ifirst:ilast]
        test_signals[i,:] = rx1
        
        dst1 = np.linalg.norm(rx1 - rx_ideal1) 
        dst[k, i] = dst1
    
    plt.figure(dpi=144)
    imax = np.argmin(dst[k,:])
    print("Minimum norm at len = %d for tau = %4.3f (noiseless)" % (template_len, tau_grid[imax]))
    plt.plot(tau_grid,dst[k,:])
    plt.title("Distances (noiseless) for len = %d" % template_len)
    plt.axvline(2.5, linestyle="dashed", color="red")

    test_signals = np.zeros((ntau, nshort),dtype='complex')
    for i, tau in enumerate(tau_grid):
        channel1 = music_tde.create_multipath_channel([tau, 11.5], [1, 0.8])
        if single_delay:
            channel1 = music_tde.create_multipath_channel([tau], [1])
        rx1 = transmit(lltf_tx, channel1, phase_noise_std=0.0, 
                              iq_gain_imbalance=0.0, 
                              iq_phase_imbalance_deg=0.0,
                              add_noise = False)
    
        rx1 = rx1[ifirst:ilast]
        test_signals[i,:] = rx1
        
        dst1 = np.linalg.norm(rx1 - rx_signal1) 
        dstn[k, i] = dst1
    
    plt.figure(dpi=144)
    imax = np.argmin(dstn[k,:])
    print("Minimum norm at len = %d for tau = %4.3f (noisy)" % (template_len, tau_grid[imax]))
    plt.plot(tau_grid,dstn[k, :])
    plt.title("Distances (noisy) for len = %d" % template_len)
    plt.axvline(2.5, linestyle="dashed", color="red")

dstm = np.copy(dst)
#for k in range(len(tem_len)):
#    dstm[k, :] = dstm[k, :] / np.max(dstm[k, :]) 
plt.figure(dpi=200)
plt.imshow(dstm, cmap='RdGy', aspect='auto', clim=(0,20))
for k in range(len(tem_len)):
    ix = np.argmin(dstm[k,:])
    plt.plot(ix,k,"o",ms=5,mec="black",mfc="black")
ix = np.where(tau_grid == 2.5)[0][0]
plt.axvline(ix)
plt.xlabel("Time delay")
plt.ylabel("Template length")
plt.colorbar()
plt.title("RMS error (ideal)")
plt.yticks(np.arange(len(tem_len)),tem_len)

dstm = np.copy(dstn)
#for k in range(len(tem_len)):
#    dstm[k, :] = dstm[k, :] / np.max(dstm[k, :]) 
plt.figure(dpi=200)
plt.imshow(dstm, cmap='RdGy', aspect='auto', clim=(0,20))
for k in range(len(tem_len)):
    ix = np.argmin(dstm[k,:])
    plt.plot(ix,k,"o",ms=5,mec="black",mfc="black")
ix = np.where(tau_grid == 2.5)[0][0]
plt.axvline(ix)
plt.xlabel("Time delay")
plt.ylabel("Template length")
plt.colorbar()
plt.title("RMS error (noisy)")
plt.yticks(np.arange(len(tem_len)),tem_len)

