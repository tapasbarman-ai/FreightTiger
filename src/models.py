from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict, Any, Tuple


@dataclass(frozen=True)
class ShipmentRecord:
    shipment_id: str
    origin: str
    destination: str
    route_type: str
    material: str
    quantity_tonnes: float
    distance_km: float
    freight_cost_inr: float
    shipment_date: date
    transporter: str

    @property
    def route(self) -> str:
        return f"{self.origin}-{self.destination}"

    @property
    def tonne_km(self) -> float:
        return self.quantity_tonnes * self.distance_km


@dataclass(frozen=True)
class ContextNote:
    note_id: str
    note_date: date
    applies_to: str
    note_text: str

    def matches_route(self, route: str) -> bool:
        if self.applies_to == "All Routes":
            return True
        return self.applies_to.strip().lower() == route.strip().lower()


@dataclass
class RouteWeeklyMetric:
    route: str
    route_type: str
    week_of: str
    total_cost: float
    total_tonne_km: float
    cost_per_tonne_km: float
    own_hist_avg: Optional[float] = None
    vs_own_history_pct: Optional[float] = None
    peer_avg: Optional[float] = None
    vs_similar_routes_pct: Optional[float] = None

    def format_own_history(self) -> str:
        if self.vs_own_history_pct is None:
            return "N/A (insufficient prior history)"
        sign = "+" if self.vs_own_history_pct >= 0 else ""
        return f"{sign}{self.vs_own_history_pct:.1f}% vs this route's past average"

    def format_similar_routes(self) -> str:
        if self.vs_similar_routes_pct is None:
            return "N/A (no peer routes)"
        sign = "+" if self.vs_similar_routes_pct >= 0 else ""
        return f"{sign}{self.vs_similar_routes_pct:.1f}% vs similar-length routes this week"


@dataclass
class AnomalyCandidate:
    metric: RouteWeeklyMetric
    is_anomaly: bool
    trigger_reason: str


@dataclass
class AuditVerdict:
    route: str
    week_of: str
    cost_per_tonne_km: float
    vs_own_history: str
    vs_similar_routes: str
    flagged: str
    matched_note_id: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "week_of": self.week_of,
            "cost_per_tonne_km": f"{self.cost_per_tonne_km:.2f}",
            "vs_own_history": self.vs_own_history,
            "vs_similar_routes": self.vs_similar_routes,
            "flagged": self.flagged,
            "matched_note_id": self.matched_note_id,
            "reason": self.reason
        }


@dataclass
class TokenCostTracker:
    model_name: str = "gpt-4o-mini"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    cost_per_million_input: float = 0.15
    cost_per_million_output: float = 0.60

    def record_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.call_count += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.prompt_tokens / 1_000_000.0) * self.cost_per_million_input
        output_cost = (self.completion_tokens / 1_000_000.0) * self.cost_per_million_output
        return input_cost + output_cost

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6)
        }


class ModelCatalog:
    RATES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "OpenAI"},
        "gpt-5.6-luna": {"input": 0.20, "output": 1.20, "provider": "OpenAI"}
    }

    @classmethod
    def get_rates(cls, model_name: str) -> Tuple[float, float]:
        normalized = model_name.lower().strip()
        if "luna" in normalized or "5.6" in normalized:
            rates = cls.RATES["gpt-5.6-luna"]
        else:
            rates = cls.RATES["gpt-4o-mini"]
        return rates["input"], rates["output"]

    @classmethod
    def compare_costs(cls, prompt_tokens: int, completion_tokens: int) -> List[Dict[str, Any]]:
        comparisons = []
        for model_key, info in cls.RATES.items():
            in_cost = (prompt_tokens / 1_000_000.0) * info["input"]
            out_cost = (completion_tokens / 1_000_000.0) * info["output"]
            tot_cost = in_cost + out_cost
            comparisons.append({
                "model": model_key,
                "provider": info["provider"],
                "input_rate_per_1m": info["input"],
                "output_rate_per_1m": info["output"],
                "input_cost_usd": round(in_cost, 6),
                "output_cost_usd": round(out_cost, 6),
                "total_cost_usd": round(tot_cost, 6)
            })
        return comparisons


if __name__ == "__main__":
    test_record = ShipmentRecord(
        shipment_id="SHP001",
        origin="Delhi",
        destination="Jaipur",
        route_type="Short",
        material="Cement",
        quantity_tonnes=20.0,
        distance_km=250.0,
        freight_cost_inr=15000.0,
        shipment_date=date(2024, 1, 1),
        transporter="Test Transport"
    )
    assert test_record.route == "Delhi-Jaipur"
    assert test_record.tonne_km == 5000.0

    test_note = ContextNote(
        note_id="N002",
        note_date=date(2025, 1, 20),
        applies_to="Ahmedabad-Mumbai",
        note_text="Festival surcharge"
    )
    assert test_note.matches_route("Ahmedabad-Mumbai") is True
    assert test_note.matches_route("Delhi-Jaipur") is False

    test_metric = RouteWeeklyMetric(
        route="Delhi-Jaipur",
        route_type="Short",
        week_of="2024-11-11",
        total_cost=41700.0,
        total_tonne_km=10000.0,
        cost_per_tonne_km=4.17,
        own_hist_avg=3.08,
        vs_own_history_pct=35.5,
        peer_avg=3.45,
        vs_similar_routes_pct=21.0
    )
    assert "+35.5%" in test_metric.format_own_history()
    assert "+21.0%" in test_metric.format_similar_routes()

    tracker = TokenCostTracker()
    tracker.record_call(150, 45)
    assert tracker.total_tokens == 195
    assert tracker.call_count == 1
    print("src/models.py: all unit assertions passed successfully.")
