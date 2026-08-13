"""Unit tests for the musicssvd public API (MUSICTDE facade).

Uses small, deterministic synthetic signals (20 MHz L-LTF, single/two-path
channels) so the suite runs fast while still exercising each TDE algorithm's
real numerical path.
"""
import numpy as np
import pytest

from musicssvd import MUSICTDE, Evaluator


@pytest.fixture
def tde():
    # High SNR, no hardware imperfections -> deterministic, easy-to-verify estimates.
    np.random.seed(0)
    return MUSICTDE(channel_width_mhz=20, snr_db=30.0, realistic_lltf=False)


def test_generate_lltf_time_domain_shape(tde):
    lltf = tde.generate_lltf_time_domain()
    # fft_size (64) * 2 symbols + cyclic prefix (16)
    assert lltf.shape == (tde.fft_size * 2 + tde.cp_len,)
    assert np.iscomplexobj(lltf)


def test_create_multipath_channel_single_path(tde):
    channel = tde.create_multipath_channel([5.0], [1.0])
    assert channel.dtype == complex
    # Peak magnitude should land near sample index 5
    peak_idx = np.argmax(np.abs(channel))
    assert abs(peak_idx - 5) <= 1


def test_create_multipath_channel_empty():
    tde = MUSICTDE(channel_width_mhz=20)
    channel = tde.create_multipath_channel([], [])
    assert channel.shape == (0,)


def test_transmit_through_channel_no_noise(tde):
    lltf_tx = tde.generate_lltf_time_domain()
    channel = tde.create_multipath_channel([2.0], [1.0])
    rx = tde.transmit_through_channel(lltf_tx, channel, add_noise=False)
    assert np.iscomplexobj(rx)
    assert len(rx) == len(lltf_tx) + len(channel) - 1


def test_cross_correlation_tde_single_path(tde):
    lltf_tx = tde.generate_lltf_time_domain()
    true_delay = 5.0
    channel = tde.create_multipath_channel([true_delay], [1.0])
    rx = tde.transmit_through_channel(lltf_tx, channel, add_noise=False)

    lags, corr, estimated = tde.cross_correlation_tde(rx, search_range_samples=(0, 20))
    assert len(lags) == len(corr)
    assert len(estimated) >= 1
    assert min(abs(e - true_delay) for e in estimated) < 1.0


def test_music_tde_two_path_resolution(tde):
    lltf_tx = tde.generate_lltf_time_domain()
    true_delays = [2.5, 11.5]
    channel = tde.create_multipath_channel(true_delays, [1.0, 0.8])
    rx = tde.transmit_through_channel(lltf_tx, channel, add_noise=False)

    tau_grid, spectrum, estimated = tde.music_tde(
        rx, num_paths=2, search_range_samples=(0, 35), subarray_size=44
    )
    assert len(tau_grid) == len(spectrum)
    assert np.isclose(np.max(spectrum), 1.0)
    assert len(estimated) == 2
    # Each true delay should have a matching estimate within 1 sample.
    for true_delay in true_delays:
        assert min(abs(true_delay - e) for e in estimated) < 1.0


def test_esprit_tde_returns_estimates(tde):
    lltf_tx = tde.generate_lltf_time_domain()
    true_delays = [2.5, 11.5]
    channel = tde.create_multipath_channel(true_delays, [1.0, 0.8])
    rx = tde.transmit_through_channel(lltf_tx, channel, add_noise=False)

    tau_grid, spectrum, estimated = tde.esprit_tde(
        rx, num_paths=2, search_range_samples=(0, 35), subarray_size=44
    )
    assert len(tau_grid) == len(spectrum)
    assert len(estimated) >= 1


def test_matrix_pencil_tde_single_path(tde):
    lltf_tx = tde.generate_lltf_time_domain()
    true_delay = 5.0
    channel = tde.create_multipath_channel([true_delay], [1.0])
    rx = tde.transmit_through_channel(lltf_tx, channel, add_noise=False)

    tau_grid, spectrum, estimated = tde.matrix_pencil_tde(
        rx, num_paths=1, search_range_samples=(0, 20)
    )
    assert len(tau_grid) == len(spectrum)
    assert len(estimated) >= 1
    assert min(abs(e - true_delay) for e in estimated) < 1.5


def test_evaluator_rmse_and_detection_rate():
    ev = Evaluator()
    true_delays = [2.5, 11.5]
    est_delays = [2.6, 11.4]

    rmse = ev.compute_rmse(true_delays, est_delays)
    assert rmse == pytest.approx(0.1, abs=1e-3)

    det_rate = ev.compute_detection_rate(true_delays, est_delays, tolerance=0.5)
    assert det_rate == 1.0


def test_evaluator_rmse_length_mismatch():
    ev = Evaluator()
    assert ev.compute_rmse([1.0], [1.0, 2.0]) == float("inf")
