import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

class Plotter:
    """
    Handles plotting of results.
    """
    def plot_results(self, true_delays: List[Union[int, float]], methods_data: dict, channel_impulse: np.ndarray = None, channel_width_mhz: int = 20):
        plt.close('all')
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        if channel_impulse is not None:
            axes[0].stem(np.arange(len(channel_impulse)), np.abs(channel_impulse))
            axes[0].set_title(f'Channel Impulse Response ({channel_width_mhz}MHz)')
            axes[0].set_xlabel('Delay (samples)')
            axes[0].set_ylabel('Magnitude')
            axes[0].grid(True, alpha=0.3)
        
        base_methods = ['Cross-Correlation', 'MUSIC', 'Multi-frame MUSIC', 'Matrix Pencil', 'Matrix Pencil MF']
        for i, method_name in enumerate(base_methods):
            ax_idx = i + 1
            if ax_idx >= len(axes):
                break
            primary = methods_data.get(method_name)
            overlay = methods_data.get(method_name + '_nonoverlap')
            if primary is None:
                continue
            tau_grid, spectrum, est_delays = primary
            axes[ax_idx].plot(tau_grid, spectrum, 'b-', linewidth=2, label='Spectrum')
            
            if overlay is not None:
                tau_o, spec_o, est_o = overlay
                axes[ax_idx].plot(tau_o, spec_o, 'm--', linewidth=1.5, label='Non-overlap')
            
            for true_delay in true_delays:
                axes[ax_idx].axvline(x=true_delay, color='r', linestyle='--', linewidth=2, label='True' if true_delay == true_delays[0] else "")
            
            for est_delay in est_delays:
                axes[ax_idx].axvline(x=est_delay, color='g', linestyle='-.', linewidth=2, label='Est' if est_delay == est_delays[0] else "")
            
            axes[ax_idx].set_title(f'{method_name} ({channel_width_mhz}MHz)')
            axes[ax_idx].set_xlabel('Delay (samples)')
            axes[ax_idx].set_ylabel('Normalized Spectrum')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
