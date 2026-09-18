import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from models import RouteWeeklyMetric, AnomalyCandidate


class AnomalyDetector:
    def __init__(self, own_history_threshold: float = 15.0, peer_route_threshold: float = 20.0):
        self.own_history_threshold = own_history_threshold
        self.peer_route_threshold = peer_route_threshold

    def evaluate_metric(self, metric: RouteWeeklyMetric) -> Tuple[bool, str]:
        own_pct = metric.vs_own_history_pct
        peer_pct = metric.vs_similar_routes_pct

        reasons = []
        if own_pct is not None and own_pct >= self.own_history_threshold:
            reasons.append(f"Own history spike (+{own_pct:.1f}% >= +{self.own_history_threshold:.1f}%)")

        if peer_pct is not None and peer_pct >= self.peer_route_threshold:
            reasons.append(f"Peer group premium (+{peer_pct:.1f}% >= +{self.peer_route_threshold:.1f}%)")

        if reasons:
            return True, "; ".join(reasons)
        return False, "Normal cost within expected bounds"

    def detect_anomalies(self, metrics: List[RouteWeeklyMetric]) -> List[AnomalyCandidate]:
        candidates: List[AnomalyCandidate] = []
        for m in metrics:
            is_anomaly, reason = self.evaluate_metric(m)
            candidates.append(
                AnomalyCandidate(
                    metric=m,
                    is_anomaly=is_anomaly,
                    trigger_reason=reason
                )
            )
        return candidates

    def get_flagged_candidates(self, metrics: List[RouteWeeklyMetric]) -> List[AnomalyCandidate]:
        all_candidates = self.detect_anomalies(metrics)
        return [c for c in all_candidates if c.is_anomaly]


if __name__ == "__main__":
    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    loader = DataLoader()
    shipments = loader.load_shipments()
    calc = MetricsCalculator()
    metrics = calc.compute_baselines(calc.aggregate_weekly_routes(shipments))

    detector = AnomalyDetector()
    anomalies = detector.get_flagged_candidates(metrics)

    print(f"src/anomaly_detector.py: Found {len(anomalies)} anomalous route-weeks.")
    target_keys = {
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
    }
    detected_keys = {(a.metric.route, a.metric.week_of) for a in anomalies}
    for tk in target_keys:
        assert tk in detected_keys, f"Missing target sample {tk} in detected anomalies"

    print("src/anomaly_detector.py: All target sample anomalies successfully detected.")
