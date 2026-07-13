# forward.py

import numpy as np
from music_tde import MUSICTDE
import matplotlib.pyplot as plt
#from typing import Tuple, Optional
# from time_varying import tv
from scipy import signal
from matplotlib.animation import FuncAnimation 

print("")

single_delay = True
show_individual = True
plot_individual = True
animate = False

template_len = 64
#template_len = 32 

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
plt.figure(dpi=300)
plt.plot(np.real(lltf_tx))
plt.title("LLTF sequence to transmit")
plt.ylabel("Re tx")
plt.ylim((-2.7,2.7))

delays_samples = [2.5, 11.5] 
#amplitudes = [1.0, 1 * np.exp(1j * 0.5), 1 * np.exp(1j * 1.2)]
amplitudes = [1, 0.8]

channel = music_tde.create_multipath_channel(delays_samples, amplitudes)
plt.figure(dpi=300)
plt.plot(np.real(channel))
plt.title("Channel")
plt.ylabel("Re IRF")
plt.ylim((-0.25,1.1))

rx_ideal = transmit(lltf_tx, channel, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)
plt.figure(dpi=300)
plt.plot(np.real(rx_ideal))
plt.title("Convoluted sequence")
plt.ylabel("Re rx")
plt.ylim((-2.7,2.7))

rx_changed = transmit(lltf_tx, channel, add_noise = False)
plt.figure(dpi=300)
plt.plot(np.real(rx_changed))
plt.title("Convoluted sequence + phase noise + gain imbalance")
plt.ylabel("Re rx")
plt.ylim((-2.7,2.7))

rx_signal = transmit(lltf_tx, channel)
plt.figure(dpi=300)
plt.plot(np.real(rx_signal))
plt.title("Received signal")
plt.ylabel("Re rx")
plt.ylim((-2.7,2.7))


channel0 = music_tde.create_multipath_channel([2.5], [1])
plt.figure(dpi=300)
plt.plot(np.real(channel0))
plt.title("Channel template ($\\tau=2.5$)")
plt.ylabel("Re IRF")
plt.ylim((-0.25,1.1))

rx_ideal0 = transmit(lltf_tx, channel0, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)
plt.figure(dpi=300)
plt.plot(np.real(rx_ideal0))
plt.title("Convoluted sequence template ($\\tau=2.5$)")
plt.ylabel("Re rx")
plt.ylim((-2.7,2.7))

channel1 = music_tde.create_multipath_channel([5.0], [1])
plt.figure(dpi=300)
plt.plot(np.real(channel1))
plt.title("Channel template ($\\tau=5.0$)")
plt.ylabel("Re IRF")
plt.ylim((-0.25,1.1))

rx_ideal1 = transmit(lltf_tx, channel1, phase_noise_std=0,
                     iq_gain_imbalance=0.0,
                     iq_phase_imbalance_deg=0.0,
                     add_noise = False)
plt.figure(dpi=300)
plt.plot(np.real(rx_ideal1))
plt.title("Convoluted sequence template ($\\tau=5.0$)")
plt.ylabel("Re rx")
plt.ylim((-2.7,2.7))



# Shorten everything
ifirst = 16
ilast = ifirst + template_len
lltf_tx1   = lltf_tx[ifirst:ilast]
rx_signal1 = rx_signal[ifirst:ilast]
rx_ideal1 = rx_ideal[ifirst:ilast]
nshort = len(lltf_tx1)

#
# define the grid
#
tau_grid = np.arange(0, 35, 0.01)
tau_grid = np.arange(0, 5, 0.01)
tau_grid = np.arange(0, 5, 0.1)
ntau = len(tau_grid)

test_signals = np.zeros((ntau, nshort),dtype='complex')
dst = np.zeros(ntau)
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
    
    dst1 = np.linalg.norm(rx1 - rx_ideal1) / np.sqrt(template_len)
    dst[i] = dst1

    if show_individual or plot_individual:
        plt.figure(dpi=200)
        plt.ylim((-3,3))
        plt.plot(np.real(rx1))
        plt.plot(np.real(rx_ideal1), linestyle="dashed", color="red")
        #plt.plot(np.real(channel1), linestyle="solid", color="orange")
        plt.title("$\\tau$ = %5.3f, dst = %5.3f" % (tau, dst1))
        #plt.axvline(2.5, color="black",linestyle="dashed")
        if plot_individual:
            plt.savefig("plots/anim1Rx0-%d-%d.png" % (template_len, i))

plt.figure(dpi=144)
imax = np.argmin(dst)
print("Minimum norm for tau = %4.3f (noiseless)" % tau_grid[imax])
plt.plot(tau_grid,dst)
plt.plot(tau_grid, dst,".",mec="black",mfc="black")
plt.title("Distances (noiseless)")
plt.axvline(tau_grid[imax], linestyle="solid", color="blue", label="Estimate")
plt.axvline(2.5, linestyle="dashed", color="red", label="Actual")
plt.legend()
plt.ylim((0,2.5))


test_signals = np.zeros((ntau, nshort),dtype='complex')
dstn = np.zeros(ntau)
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
    
    dst1 = np.linalg.norm(rx1 - rx_signal1) / np.sqrt(template_len)
    dstn[i] = dst1

    if show_individual or plot_individual:
        plt.figure(dpi=200)
        plt.ylim((-3,3))
        plt.plot(np.real(rx1))
        plt.plot(np.real(rx_signal1), linestyle="dashed", color="red")
        #plt.plot(np.real(channel1), linestyle="solid", color="orange")
        plt.title("$\\tau$ = %5.3f, dst = %5.3f" % (tau, dst1))
        #plt.axvline(2.5, color="black",linestyle="dashed")
        if plot_individual:
            plt.savefig("plots/anim1Rx1-%d-%d.png" % (template_len, i))

plt.figure(dpi=144)
imax = np.argmin(dstn)
print("Minimum norm for tau = %4.3f (noisy)" % tau_grid[imax])
plt.plot(tau_grid,dstn)
plt.plot(tau_grid, dstn,".",mec="black",mfc="black")
plt.title("Distances (noisy)")
plt.axvline(tau_grid[imax], linestyle="solid", color="blue", label="Estimate")
plt.axvline(2.5, linestyle="dashed", color="red", label="Actual")
plt.legend()
plt.ylim((0,2.5))

if animate:
    print("")
    print("Animating...")
    for k in [0, 1]:
        fig = plt.figure(dpi=144) 
        # marking the x-axis and y-axis
        axis = plt.axes(xlim =(0, 64), 
                        ylim =(-3, 3)) 
        line, = axis.plot([], [], lw = 2)  
        if k == 0:
            plt.plot(np.real(rx_ideal1), linestyle="dashed", color="red")
        if k == 1:
            plt.plot(np.real(rx_signal1), linestyle="dashed", color="green")
        def init(): 
            line.set_data([], [])
            return line, 
        def animate(i):
            
            tau = tau_grid[i]
            channel1 = music_tde.create_multipath_channel([tau, 11.5], [1, 0.8])
            if single_delay:
                channel1 = music_tde.create_multipath_channel([tau], [1])
            rx1 = transmit(lltf_tx, channel1, phase_noise_std=0.0, 
                                  iq_gain_imbalance=0.0, 
                                  iq_phase_imbalance_deg=0.0,
                                  add_noise = False)
            rx1 = rx1[ifirst:ilast]
            #plt.plot(np.real(rx1))
            if k == 0:
                dst1 = np.linalg.norm(rx1 - rx_ideal1) / np.sqrt(template_len)
            if k == 1:
                dst1 = np.linalg.norm(rx1 - rx_signal1)  / np.sqrt(template_len)
            plt.title("$\\tau$ = %5.3f, dst = %5.3f" % (tau, dst1))
            x = np.linspace(0,64,64)
            y = np.real(rx1)
            line.set_data(x, y)
            return line,
        anim = FuncAnimation(fig, animate, init_func = init,
                             frames = len(tau_grid), interval = 200, blit = True)
        fname = "anim2Rx%d.mp4" % k
        if single_delay:
            fname = "anim1Rx%d.mp4" % k        
        anim.save(fname, writer = 'ffmpeg', fps = 25)
    
    
    
    
