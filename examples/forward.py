# forward.py

import numpy as np
from musicssvd import MUSICTDE
import matplotlib.pyplot as plt
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

transmit = music_tde.transmit_through_channel

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
    
    
    
    
