# basic_usage.py
#
# Minimal example of the musicssvd API: generate an L-LTF signal, pass it
# through a synthetic two-path channel, and estimate the path delays with
# MUSIC. No plotting, no side experiments.

from musicssvd import MUSICTDE

music_tde = MUSICTDE(channel_width_mhz=20, snr_db=20.0)

lltf_tx = music_tde.generate_lltf_time_domain()

true_delays_samples = [2.5, 11.5]
channel = music_tde.create_multipath_channel(true_delays_samples, amplitudes=[1.0, 0.8])

rx_signal = music_tde.transmit_through_channel(lltf_tx, channel)

num_paths = len(true_delays_samples)
tau_grid, spectrum, estimated_delays = music_tde.music_tde(
    rx_signal, num_paths, search_range_samples=(0, 35), subarray_size=44
)

print(f"True delays:      {true_delays_samples}")
print(f"Estimated delays: {estimated_delays}")
