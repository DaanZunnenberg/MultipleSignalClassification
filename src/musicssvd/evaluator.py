import numpy as np
from typing import List, Dict

class Evaluator:
    """
    Handles evaluation metrics for TDE results.
    """
    def compute_rmse(self, true_delays: List[float], est_delays: List[float]) -> float:
        if len(true_delays) != len(est_delays):
            return float('inf')  # Mismatch
        return np.sqrt(np.mean((np.array(true_delays) - np.array(est_delays))**2))
    
    def compute_detection_rate(self, true_delays: List[float], est_delays: List[float], tolerance: float = 1.0) -> float:
        detected = 0
        for true in true_delays:
            if any(abs(true - est) <= tolerance for est in est_delays):
                detected += 1
        return detected / len(true_delays) if true_delays else 0.0
    
    def evaluate_all(self, methods_data: Dict, true_delays: List[float]) -> Dict:
        results = {}
        for method, (tau_grid, spectrum, est_delays) in methods_data.items():
            if est_delays:
                rmse = self.compute_rmse(true_delays, est_delays)
                det_rate = self.compute_detection_rate(true_delays, est_delays)
                results[method] = {'rmse': rmse, 'detection_rate': det_rate}
        return results
