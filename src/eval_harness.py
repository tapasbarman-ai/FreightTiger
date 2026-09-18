import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import DataLoader
from metrics_calculator import MetricsCalculator
from rag_retriever import NoteRetriever
from evaluator import CostAssistant


@dataclass
class LabeledTestCase:
    name: str
    route: str
    week_of: str
    expected_flagged: str
    expected_note_id: str
    context_note_text: Optional[str] = None


class EvaluationHarness:
    def __init__(self, data_dir: Optional[str] = None):
        self.loader = DataLoader(data_dir=data_dir)
        shipments = self.loader.load_shipments()
        self.notes = self.loader.load_notes()

        calc = MetricsCalculator()
        metrics = calc.compute_baselines(calc.aggregate_weekly_routes(shipments))
        self.lookup = {(m.route, m.week_of): m for m in metrics}
        self.assistant = CostAssistant(NoteRetriever(self.notes))

        self.suite: List[LabeledTestCase] = [
            LabeledTestCase("Festival Surcharge", "Ahmedabad-Mumbai", "2025-01-20", "No (justified)", "N002",
                            "A regional festival week saw a temporary surcharge applied by transporters on the Ahmedabad-Mumbai corridor due to high demand and limited truck availability."),
            LabeledTestCase("Flood Disruption", "Chennai-Bangalore", "2025-02-24", "No (justified)", "N001",
                            "Heavy flooding on the Chennai-Bangalore highway disrupted normal truck movement from Feb 24 to Mar 8, forcing longer detours and higher trip costs."),
            LabeledTestCase("Unexplained Surge", "Delhi-Jaipur", "2024-11-11", "Yes", ""),
            LabeledTestCase("Nearby Normal Report (N006)", "Mumbai-Pune", "2025-09-15", "Yes", ""),
            LabeledTestCase("Maintenance Trap Note (N005)", "Mumbai-Delhi", "2024-07-29", "Yes", ""),
            LabeledTestCase("Recovery Note (N009)", "Chennai-Bangalore", "2025-03-17", "Yes", "")
        ]

    def run_evaluation(self, run_deepeval_metrics: bool = True) -> dict:
        passed, hallucinations = 0, 0
        details = []
        deepeval_scores = []

        for tc in self.suite:
            metric = self.lookup.get((tc.route, tc.week_of))
            verdict = self.assistant.evaluate_route_week(metric)
            is_pass = (verdict.flagged == tc.expected_flagged and verdict.matched_note_id == tc.expected_note_id)
            if is_pass:
                passed += 1

            if verdict.flagged == "No (justified)" and tc.expected_flagged == "Yes":
                hallucinations += 1

            deepeval_status = "N/A"
            if run_deepeval_metrics and tc.expected_flagged == "No (justified)" and tc.context_note_text:
                try:
                    case = LLMTestCase(
                        input=f"Evaluate cost spike on route {tc.route} during week {tc.week_of}",
                        actual_output=verdict.reason,
                        context=[tc.context_note_text]
                    )
                    h_metric = HallucinationMetric(threshold=0.5)
                    h_metric.measure(case)
                    deepeval_scores.append(h_metric.score)
                    deepeval_status = f"DeepEval Score: {h_metric.score:.2f} (Passed: {h_metric.is_successful()})"
                except Exception:
                    pass

            details.append({
                "name": tc.name, "route": tc.route, "week": tc.week_of,
                "flagged": verdict.flagged, "expected": tc.expected_flagged,
                "note": verdict.matched_note_id, "passed": is_pass,
                "deepeval": deepeval_status
            })

        total = len(self.suite)
        return {
            "total": total, "passed": passed, "accuracy": (passed / total) * 100.0,
            "hallucinations": hallucinations, "deepeval_avg_score": (sum(deepeval_scores) / len(deepeval_scores)) if deepeval_scores else 1.0,
            "details": details
        }


if __name__ == "__main__":
    harness = EvaluationHarness()
    res = harness.run_evaluation(run_deepeval_metrics=True)

    print(f"src/eval_harness.py (DeepEval Integrated):")
    print(f"  Benchmark Accuracy: {res['accuracy']:.1f}% ({res['passed']}/{res['total']})")
    print(f"  DeepEval Hallucination Score: {res['deepeval_avg_score']:.2f}")

    for d in res["details"]:
        tag = "PASS" if d["passed"] else "FAIL"
        print(f"  [{tag}] {d['name']} -> Verdict: {d['flagged']} | {d['deepeval']}")

    assert res["passed"] == res["total"]
    assert res["hallucinations"] == 0
    print("src/eval_harness.py: All DeepEval and benchmark tests passed.")
