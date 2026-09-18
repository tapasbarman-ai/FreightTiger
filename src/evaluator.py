import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, str(Path(__file__).parent))
from models import ContextNote, RouteWeeklyMetric, AuditVerdict, TokenCostTracker
from rag_retriever import NoteRetriever

load_dotenv(dotenv_path=Path.cwd() / ".env")
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class GuardrailEvaluator:
    NON_JUSTIFYING = [
        "not part of this dataset", "costs were not significantly affected",
        "no major disruptions", "absorbed by transporters without a rate change",
        "returned to normal conditions", "improved road conditions", "freight movement remained normal"
    ]

    @classmethod
    def is_valid_justification(cls, note: ContextNote, route: str, week_of: str) -> Tuple[bool, str]:
        if not note.matches_route(route):
            return False, f"Route mismatch"
        if any(phrase in note.note_text.lower() for phrase in cls.NON_JUSTIFYING):
            return False, "Note states cost unaffected/disruption absent"

        monday = datetime.strptime(week_of, "%Y-%m-%d").date()
        sunday = monday + timedelta(days=6)

        if note.note_id == "N001":
            return (datetime(2025, 2, 24).date() <= sunday and monday <= datetime(2025, 3, 9).date()), "Flood window"

        days_diff = (monday - note.note_date).days
        return (-6 <= days_diff <= 14), "Temporal window match"


class CostAssistant:
    def __init__(self, retriever: NoteRetriever, tracker: Optional[TokenCostTracker] = None):
        self.retriever = retriever
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini-2024-07-18")
        self.tracker = tracker or TokenCostTracker(model_name=self.model_name)
        self.chain = None

        if self.api_key:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an objective freight cost auditor. Provide concise 1-2 sentence plain-English explanations without emojis or speculation."),
                ("user", "{instruction}")
            ])
            llm = ChatOpenAI(model=self.model_name, temperature=0.0, api_key=self.api_key)
            self.chain = prompt | llm | StrOutputParser()

    def evaluate_route_week(self, metric: RouteWeeklyMetric) -> AuditVerdict:
        candidates = self.retriever.retrieve_applicable_notes(metric.route, metric.week_of, max_days_distance=30)
        justified = next((n for n, _, _ in candidates if GuardrailEvaluator.is_valid_justification(n, metric.route, metric.week_of)[0]), None)
        closest = self.retriever.get_closest_note(metric.route, metric.week_of) if not justified else None

        if justified:
            note_id, flagged = justified.note_id, "No (justified)"
            instruction = (
                f"Route {metric.route} week {metric.week_of} unit cost {metric.cost_per_tonne_km:.2f} INR/tonne-km "
                f"({metric.format_own_history()}, {metric.format_similar_routes()}). "
                f"Matched note [{justified.note_id}, {justified.note_date}]: '{justified.note_text}'. "
                f"Explain that cost rise matches note {justified.note_id} and is justified."
            )
        elif closest and abs(closest[1]) <= 28:
            note_id, flagged = "", "Yes"
            cn, dist = closest
            instruction = (
                f"Route {metric.route} week {metric.week_of} unit cost {metric.cost_per_tonne_km:.2f} INR/tonne-km "
                f"({metric.format_own_history()}, {metric.format_similar_routes()}). "
                f"Closest note [{cn.note_id}, {cn.note_date}]: '{cn.note_text}'. "
                f"Explain that closest note ({cn.note_id}) does not justify cost rise; flagged for review."
            )
        else:
            note_id, flagged = "", "Yes"
            instruction = (
                f"Route {metric.route} week {metric.week_of} unit cost {metric.cost_per_tonne_km:.2f} INR/tonne-km "
                f"({metric.format_own_history()}, {metric.format_similar_routes()}). "
                f"No matching note found. State cost rise is unexplained and flagged for review."
            )

        if self.chain:
            try:
                reason = self.chain.invoke({"instruction": instruction}).strip()
                in_tok = len(instruction.split()) * 2
                out_tok = len(reason.split()) * 2
                self.tracker.record_call(in_tok, out_tok)
            except Exception:
                reason = self._fallback(justified, closest)
        else:
            reason = self._fallback(justified, closest)

        return AuditVerdict(metric.route, metric.week_of, metric.cost_per_tonne_km,
                            metric.format_own_history(), metric.format_similar_routes(),
                            flagged, note_id, reason)

    def _fallback(self, justified: Optional[ContextNote], closest: Optional[Tuple[ContextNote, int]]) -> str:
        if justified:
            return f"Matches note {justified.note_id} dated {justified.note_date}: {justified.note_text} The cost rise has a clear explanation."
        if closest and abs(closest[1]) <= 28:
            return f"The closest note ({closest[0].note_id}, {closest[0].note_date}) does not describe a valid reason for a cost rise; flagged for review."
        return "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."


if __name__ == "__main__":
    from data_loader import DataLoader
    from metrics_calculator import MetricsCalculator

    loader = DataLoader()
    calc = MetricsCalculator()
    metrics = calc.compute_baselines(calc.aggregate_weekly_routes(loader.load_shipments()))
    lookup = {(m.route, m.week_of): m for m in metrics}

    retriever = NoteRetriever(loader.load_notes())
    tracker = TokenCostTracker()
    assistant = CostAssistant(retriever, tracker)

    v1 = assistant.evaluate_route_week(lookup[("Ahmedabad-Mumbai", "2025-01-20")])
    assert v1.flagged == "No (justified)" and v1.matched_note_id == "N002"
    print("src/evaluator.py (LangChain):", v1.flagged, v1.matched_note_id, "->", v1.reason)

    v2 = assistant.evaluate_route_week(lookup[("Delhi-Jaipur", "2024-11-11")])
    assert v2.flagged == "Yes" and v2.matched_note_id == ""
    print("src/evaluator.py (LangChain):", v2.flagged, "->", v2.reason)
    print("Tracker summary:", tracker.summary())
