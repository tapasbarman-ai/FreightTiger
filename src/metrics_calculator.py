import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from models import ShipmentRecord, RouteWeeklyMetric


class MetricsCalculator:
    def __init__(self, rolling_window_weeks: int = 8):
        self.rolling_window_weeks = rolling_window_weeks

    @staticmethod
    def get_monday_of_week(record_date) -> str:
        monday = record_date - timedelta(days=record_date.weekday())
        return monday.strftime("%Y-%m-%d")

    def aggregate_weekly_routes(self, shipments: List[ShipmentRecord]) -> List[RouteWeeklyMetric]:
        grouped_data: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
            lambda: {"total_cost": 0.0, "total_tonne_km": 0.0}
        )

        for s in shipments:
            week_of = self.get_monday_of_week(s.shipment_date)
            key = (s.route, s.route_type, week_of)
            grouped_data[key]["total_cost"] += s.freight_cost_inr
            grouped_data[key]["total_tonne_km"] += s.tonne_km

        metrics: List[RouteWeeklyMetric] = []
        for (route, route_type, week_of), vals in grouped_data.items():
            tot_cost = vals["total_cost"]
            tot_tkm = vals["total_tonne_km"]
            unit_cost = tot_cost / tot_tkm if tot_tkm > 0 else 0.0
            metrics.append(
                RouteWeeklyMetric(
                    route=route,
                    route_type=route_type,
                    week_of=week_of,
                    total_cost=tot_cost,
                    total_tonne_km=tot_tkm,
                    cost_per_tonne_km=unit_cost
                )
            )

        metrics.sort(key=lambda m: (m.route, m.week_of))
        return metrics

    def compute_baselines(self, metrics: List[RouteWeeklyMetric]) -> List[RouteWeeklyMetric]:
        route_series: Dict[str, List[RouteWeeklyMetric]] = defaultdict(list)
        for m in metrics:
            route_series[m.route].append(m)

        for route, series in route_series.items():
            series.sort(key=lambda m: m.week_of)
            costs = [m.cost_per_tonne_km for m in series]
            for i, m in enumerate(series):
                start_idx = max(0, i - self.rolling_window_weeks)
                prior_slice = costs[start_idx:i]
                if prior_slice:
                    avg_prior = sum(prior_slice) / len(prior_slice)
                    pct_diff = ((m.cost_per_tonne_km - avg_prior) / avg_prior) * 100.0
                    m.own_hist_avg = avg_prior
                    m.vs_own_history_pct = pct_diff
                else:
                    m.own_hist_avg = None
                    m.vs_own_history_pct = None

        week_peer_groups: Dict[Tuple[str, str], List[RouteWeeklyMetric]] = defaultdict(list)
        for m in metrics:
            week_peer_groups[(m.week_of, m.route_type)].append(m)

        for (week_of, route_type), group in week_peer_groups.items():
            for m in group:
                peers = [p for p in group if p.route != m.route]
                if peers:
                    peer_avg = sum(p.cost_per_tonne_km for p in peers) / len(peers)
                    pct_diff = ((m.cost_per_tonne_km - peer_avg) / peer_avg) * 100.0
                    m.peer_avg = peer_avg
                    m.vs_similar_routes_pct = pct_diff
                else:
                    m.peer_avg = None
                    m.vs_similar_routes_pct = None

        metrics.sort(key=lambda m: (m.route, m.week_of))
        return metrics


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    shipments = loader.load_shipments()

    calculator = MetricsCalculator()
    weekly_metrics = calculator.aggregate_weekly_routes(shipments)
    computed = calculator.compute_baselines(weekly_metrics)

    metric_lookup = {(m.route, m.week_of): m for m in computed}

    m1 = metric_lookup.get(("Delhi-Jaipur", "2024-11-11"))
    assert m1 is not None
    assert round(m1.cost_per_tonne_km, 2) == 4.17
    assert round(m1.vs_own_history_pct, 1) == 35.5
    assert round(m1.vs_similar_routes_pct, 1) == 21.0

    m2 = metric_lookup.get(("Ahmedabad-Mumbai", "2025-01-20"))
    assert m2 is not None
    assert round(m2.cost_per_tonne_km, 2) == 3.29
    assert round(m2.vs_own_history_pct, 1) == 29.5
    assert round(m2.vs_similar_routes_pct, 1) == 22.5

    m3 = metric_lookup.get(("Mumbai-Pune", "2025-09-15"))
    assert m3 is not None
    assert round(m3.cost_per_tonne_km, 2) == 3.98
    assert round(m3.vs_own_history_pct, 1) == 9.2
    assert round(m3.vs_similar_routes_pct, 1) == 23.6

    print("src/metrics_calculator.py: All baseline calculations matched exact sample specifications.")
