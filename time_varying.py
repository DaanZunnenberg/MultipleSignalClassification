# time_varying.py


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema, find_peaks

from music_tde import MUSICTDE

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


clight = 3E8
lam = clight / 2.4E9  # 12.5 cm
# for coherence 0.90 need to go down to 1/15 of this, i.e., <1.0 cm

#vels = np.linspace(-1,1,128)
taus = np.linspace(0,32,64)
ts   = np.linspace(0,5,128)

# Doppler formula: fv = 2*v / lam 
dtau = np.diff(taus)[0]
ntau = len(taus)
nt = len(ts)

# Time delay for samples / Frequency
ftaus = 20E6  # sample rate
dtau = 1/ftaus  # 50 ns 
freqstau = np.fft.fftfreq(ntau)*ftaus # up to 10 MHz possible
freqstau = np.fft.fftshift(freqstau)
dfreqtau = np.diff(freqstau)[0]        # 20 MHz / nt

# Time for motion / Doppler frequency
# velocity <-> [ Doppler frequency <-> time ]
dt = np.diff(ts)[0]  # e.g. 40 ms for 128 samples in 5 seconds
fs = 1/dt            # e.g. 25.4 Hz
freqs = np.fft.fftfreq(nt)*fs  # possible Doppler frequencies, up to 12.7 Hz
freqs = np.fft.fftshift(freqs)
dfreq = np.diff(freqs)[0]      # e.g. 0.4 Hz
vs = lam * freqs / 2           # possible velocities, up to e.g. 0.8 m/s
nv = len(vs)




save_plots = True
individual_plots = True
save_plots = False
individual_plots = False

lltf_tx = music_tde.generate_lltf_time_domain()

def rmse(rx, tau, template_len=64, ifirst=16):
    channel1 = music_tde.create_multipath_channel([tau], [1])
    rx1 = transmit(lltf_tx, channel1, phase_noise_std=0.0, 
                          iq_gain_imbalance=0.0, 
                          iq_phase_imbalance_deg=0.0,
                          add_noise = False)
    ilast = ifirst + template_len 
    rx1 = rx1[ifirst:ilast]
    rx  = rx[ifirst:ilast]
    dst = np.linalg.norm(rx1 - rx) / np.sqrt(template_len) 
    return dst

def find_tau(channel, ifirst=16):
    rx = transmit(lltf_tx, channel, phase_noise_std=0.0, 
                          iq_gain_imbalance=0.0, 
                          iq_phase_imbalance_deg=0.0,
                          add_noise = True)
    tau_grid = np.arange(0,20.0,0.01)
    ntau = len(tau_grid)
    dst = np.zeros(ntau)
    for i, tau in enumerate(tau_grid):
        err = rmse(rx, tau, template_len=128, ifirst=ifirst)
        dst[i] = err
    imin = np.argmin(dst)
    taumin = tau_grid[imin]
    return taumin




def fgaussian(x, loc = 0.0, scale = 0.1):
    f = 1/np.sqrt(2*np.pi)/scale * np.exp(-0.5*((x-loc)/scale)**2)
    return f

def fsinc(x, loc = 0.0, scale = 0.5):
    f = np.sinc(scale * (loc-x)) / scale 
    return f 

def localize(S, t, v, scalet=0.5, scalev=0.1):
    # Doppler formula: fv = 2*v / lam 
    fv = 2*v/lam 
    Sadd = np.zeros_like(S) 
    # lazy
    for i in range(len(taus)):
        for j in range(len(freqs)):
            Sadd[i,j] = fsinc(taus[i], loc=t, scale=scalet) * fgaussian(freqs[j], loc=fv, scale=scalev)            
    return S + Sadd

def find_delays(channel1, thresh=0.005):
    x = np.abs(channel1)
    # threshold
    x[x < thresh] = 0.0
    ix = find_peaks(x)[0]
    peaks = x[ix]
    ip = np.argsort(peaks)
    ip = np.flip(ip)
    peaks = peaks[ip]
    ix = ix[ip]
    return ix 


#def demonstrate_tv():
S = np.zeros((ntau, nv))
S = localize(S, 3.2, 0.1, scalet = 0.7, scalev = 0.2)   
S = localize(S, 17.0, -0.05, scalet = 1.0, scalev = 0.5)   
S = localize(S, 7.5, 0.0, scalet = 1.0, scalev = 0.1)   

plt.figure(dpi=600)
#plt.imshow(S.T)
X, Y = np.meshgrid(taus, freqs)
Z = S.T
#plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.imshow(Z, extent=[taus[0], taus[-1], freqs[0], freqs[-1]], origin='lower', cmap='Oranges', aspect='auto')
plt.colorbar()
plt.xticks(np.arange(0,32,5),np.arange(0,32,5))  # samples
plt.yticks()
plt.xlabel("$\\tau$ [50 ns samples]")
plt.ylabel("Doppler frequency [Hz]")
        
fig = plt.figure(dpi=600)
ax = fig.add_subplot(projection='3d')        
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1)    
plt.xlabel("$\\tau$")
plt.ylabel("$Doppler$")
    
# iFFT on Doppler freqs
B = np.fft.ifft(S, axis=1) 
B = np.fft.ifftshift(B, axes=1)

plt.figure(dpi=600)
X, Y = np.meshgrid(taus, freqs)
Z = np.abs(B.T)
plt.imshow(Z, extent=[taus[0], taus[-1], ts[0], ts[-1]], origin='lower', cmap='RdGy', aspect='auto')
plt.colorbar()
plt.xlabel("$\\tau$ [50 ns samples]")
plt.ylabel("$t$ [s]")

plt.figure(dpi=600)
X, Y = np.meshgrid(taus, freqs)
logZ = 20*np.log10(Z)
plt.imshow(logZ, extent=[taus[0], taus[-1], ts[0], ts[-1]], origin='lower', cmap='RdGy', aspect='auto')
plt.colorbar()
plt.xlabel("$\\tau$ [50 ns samples]")
plt.ylabel("$t$ [s]")

fig = plt.figure(dpi=600)
ax = fig.add_subplot(projection='3d')        
#ax.plot_surface(X, Y, Z, cmap="terrain")
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, linewidth=1)    
plt.xlabel("$\\tau$")
plt.ylabel("$t$ [s]")

lltf_tx = music_tde.generate_lltf_time_domain()

# plot slices
smin = 0
slen = 128
smax = smin + slen
k = 0
btaus = np.zeros(slen)
bts   = np.zeros(slen)
for i in range(smin, smax):
    channel1 = Z[i,:]
    tau1 = find_tau(channel1)
    btaus[k] = tau1
    bts[k] = ts[i]
    print("i = ", i, " tau1 = ", tau1)
    
    rx1 = transmit(lltf_tx, channel1, phase_noise_std=0.0, 
                          iq_gain_imbalance=0, iq_phase_imbalance_deg=0,
                          add_noise = False)
    if individual_plots or save_plots:
        plt.figure(dpi=144)
        plt.title("Channel - slice %d" % i)
        #plt.plot(taus,np.log10(Z[i,:]))
        #plt.ylim((-7,0.5))
        plt.plot(channel1,label="Channel")
        plt.ylim((0,0.1))
        plt.title("t = %4.3fs" % ts[i])
        plt.xlabel("$\\tau$")
        plt.yticks([])
        plt.plot(np.linspace(0,42,len(rx1)),np.real(rx1/5 + 0.05), color="red", label="Signal (ideal)")
        plt.legend()
        plt.xlim((0,45))
        
        if save_plots:
            plt.savefig("plots/tv-channel-%d" % k)
    
    k = k + 1


plt.figure(dpi=200)
plt.ylabel("Time [s]")
plt.xlabel("$\\tau$ [50 ns samples]")
plt.xlim((0,20.0))
plt.title("Peaks (RGB)")
#plt.axhline(7.5,linestyle="dashed",color="grey")
#plt.axhline(17.0,linestyle="dashed",color="grey")
#plt.axhline(3.2,linestyle="dashed",color="grey")
k = 0
colors = ["red", "green", "blue", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey"]
for i in range(smin, smax):
    channel1 = Z[i,:]
    ix = find_delays(channel1)
    print(i, "t = %4.3f" % bts[k])
    for j, tau in enumerate(ix):
        plt.plot(tau, bts[k], "o", ms=2, color=colors[j])
        print("  ix = %4.3f" % tau)
    k = k + 1

plt.figure(dpi=200)
plt.plot(btaus, bts)
plt.ylabel("Time [s]")
plt.xlabel("$\\tau$ [50 ns samples]")
plt.xlim((0,20.0))
plt.title("Peak detection")
#plt.axhline(7.5,linestyle="dashed",color="grey")
#plt.axhline(17.0,linestyle="dashed",color="grey")
#plt.axhline(3.2,linestyle="dashed",color="grey")
k = 0
colors = ["red", "green", "blue", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey"]
for i in range(smin, smax):
    channel1 = Z[i,:]
    ix = find_delays(channel1)
    print(i, "t = %4.3f" % bts[k])
    for j, tau in enumerate(ix):
        plt.plot(tau, bts[k], "o", ms=2, color=colors[j])
        print("  ix = ", tau)
    #if i == 96:
    #    plt.axvline(bts[k])
    #    print("*** i = ", i, "   k = ", k, "***")
    #if i == 125:
    #    plt.axvline(bts[k])
    #    print("*** i = ", i, "   k = ", k, "***")
    k = k + 1
plt.plot(btaus, bts, "o", color="black", ms=6, markerfacecolor='none')
    
#return S, Z



#if __name__ == "__main__":
#    S, Z = demonstrate_tv()
