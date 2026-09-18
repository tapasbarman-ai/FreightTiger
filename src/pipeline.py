import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from models import AuditVerdict, TokenCostTracker
from data_loader import DataLoader
from metrics_calculator import MetricsCalculator
from anomaly_detector import AnomalyDetector
from rag_retriever import NoteRetriever
from evaluator import CostAssistant


class ShippingCostPipeline:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        own_history_threshold: float = 15.0,
        peer_route_threshold: float = 20.0,
        model_name: str = "gpt-4o-mini"
    ):
        self.loader = DataLoader(data_dir=data_dir)
        self.calculator = MetricsCalculator(rolling_window_weeks=8)
        self.detector = AnomalyDetector(
            own_history_threshold=own_history_threshold,
            peer_route_threshold=peer_route_threshold
        )
        self.model_name = model_name

    def run(self) -> Tuple[List[AuditVerdict], TokenCostTracker]:
        shipments = self.loader.load_shipments()
        notes = self.loader.load_notes()

        aggregated_metrics = self.calculator.aggregate_weekly_routes(shipments)
        computed_metrics = self.calculator.compute_baselines(aggregated_metrics)

        anomalies = self.detector.get_flagged_candidates(computed_metrics)

        retriever = NoteRetriever(notes)
        tracker = TokenCostTracker(model_name=self.model_name)
        assistant = CostAssistant(retriever, tracker)

        verdicts: List[AuditVerdict] = []
        for anomaly in anomalies:
            verdict = assistant.evaluate_route_week(anomaly.metric)
            verdicts.append(verdict)

        verdicts.sort(key=lambda v: (v.route, v.week_of))
        return verdicts, tracker

    def export_csv(self, verdicts: List[AuditVerdict], output_filepath: str = "output_submission.csv") -> None:
        out_path = Path(output_filepath)
        if not out_path.is_absolute():
            # If relative, ensure parent dir exists
            if out_path.parent:
                out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "route",
            "week_of",
            "cost_per_tonne_km",
            "vs_own_history",
            "vs_similar_routes",
            "flagged",
            "matched_note_id",
            "reason"
        ]

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in verdicts:
                writer.writerow(v.to_dict())

        # Also mirror to output/ if outputting to root, or root if outputting to output/
        alt_path = Path("output") / out_path.name if out_path.parent == Path(".") else Path(out_path.name)
        if alt_path != out_path:
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(alt_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for v in verdicts:
                    writer.writerow(v.to_dict())

    def verify_reproducibility(self, runs: int = 3) -> bool:
        previous_results = None
        for run_idx in range(1, runs + 1):
            verdicts, _ = self.run()
            serialized = [
                (v.route, v.week_of, f"{v.cost_per_tonne_km:.2f}", v.vs_own_history, v.vs_similar_routes, v.flagged, v.matched_note_id)
                for v in verdicts
            ]
            if previous_results is not None:
                if serialized != previous_results:
                    return False
            previous_results = serialized
        return True


if __name__ == "__main__":
    pipeline = ShippingCostPipeline()
    verdicts, tracker = pipeline.run()
    pipeline.export_csv(verdicts, "output_submission.csv")

    print(f"src/pipeline.py: Processed {len(verdicts)} anomalous route-weeks.")
    print("src/pipeline.py: Token tracker summary:", tracker.summary())

    reproducible = pipeline.verify_reproducibility(runs=3)
    assert reproducible is True
    print("src/pipeline.py: 3 consecutive runs confirmed 100% deterministic reproducibility.")
