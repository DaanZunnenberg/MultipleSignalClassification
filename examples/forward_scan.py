# forward_scan.py

import numpy as np
from musicssvd import MUSICTDE
import matplotlib.pyplot as plt

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

transmit = music_tde.transmit_through_channel

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

