import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ShippingCostPipeline
from eval_harness import EvaluationHarness
from models import AuditVerdict, ModelCatalog


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class AssistantQAService:
    def __init__(self, verdicts: List[AuditVerdict]):
        self.verdicts = verdicts
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("MODEL_NAME", "gpt-4o-mini-2024-07-18")
        self.chain = None
        if api_key:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are Freight Tiger's Shipping Cost Assistant. Answer the user's question directly and factually based ONLY on the provided route findings. Be concise (2-3 sentences), professional, and do not use emojis."),
                ("user", "Findings:\n{context}\n\nQuestion: {question}")
            ])
            self.chain = prompt | ChatOpenAI(model=model, temperature=0.0, api_key=api_key) | StrOutputParser()

    def answer_question(self, question: str) -> str:
        q_lower = question.lower()
        matched = [v for v in self.verdicts if any(p in q_lower for p in v.route.lower().split("-")) or v.route.lower() in q_lower]
        target = matched if matched else self.verdicts[:5]
        context = "\n".join(
            f"- Route: {v.route}, Week: {v.week_of}, Cost: {v.cost_per_tonne_km:.2f} ({v.vs_own_history}, {v.vs_similar_routes}), "
            f"Flagged: {v.flagged}, Matched Note: {v.matched_note_id}, Reason: {v.reason}"
            for v in target
        )

        if self.chain:
            try:
                return self.chain.invoke({"context": context, "question": question}).strip()
            except Exception:
                pass

        return "\n\n".join(f"On route {v.route} ({v.week_of}), cost {v.cost_per_tonne_km:.2f} INR/tonne-km. Reason: {v.reason}" for v in target)


class ShippingCostCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Freight Tiger Shipping Cost Assistant CLI")
        self.setup_arguments()

    def setup_arguments(self):
        subparsers = self.parser.add_subparsers(dest="command")

        subparsers.add_parser("run", help="Run full pipeline and generate output CSV")
        subparsers.add_parser("verify", help="Run 3 consecutive reproducibility passes")
        subparsers.add_parser("eval", help="Run anti-hallucination benchmark evaluation harness")
        subparsers.add_parser("compare", help="Compare inference costs between GPT-4o-mini and GPT-5.6 Luna")

        ask_parser = subparsers.add_parser("ask", help="Ask plain-English questions about findings")
        ask_parser.add_argument("question", type=str, help="Question to ask the assistant")

    def execute(self, args: Optional[List[str]] = None):
        parsed = self.parser.parse_args(args)
        cmd = parsed.command or "run"

        if cmd == "run":
            pipeline = ShippingCostPipeline()
            verdicts, tracker = pipeline.run()
            output_file = "output_submission.csv"
            pipeline.export_csv(verdicts, output_file)
            print(f"Execution complete. Successfully wrote {len(verdicts)} rows to {output_file}")
            print("Token and cost summary:")
            for k, val in tracker.summary().items():
                print(f"  {k}: {val}")

        elif cmd == "compare":
            pipeline = ShippingCostPipeline()
            verdicts, tracker = pipeline.run()
            from models import ModelCatalog
            comparisons = ModelCatalog.compare_costs(tracker.prompt_tokens, tracker.completion_tokens)
            print("\n" + "=" * 80)
            print(f"MODEL COST COMPARISON ({tracker.call_count} calls | {tracker.prompt_tokens} prompt tokens | {tracker.completion_tokens} completion tokens)")
            print("=" * 80)
            print(f"{'Model':<16} | {'Provider':<18} | {'In/1M':<8} | {'Out/1M':<8} | {'Total Cost USD'}")
            print("-" * 80)
            for c in comparisons:
                print(f"{c['model']:<16} | {c['provider']:<18} | ${c['input_rate_per_1m']:<7.2f} | ${c['output_rate_per_1m']:<7.2f} | ${c['total_cost_usd']:.6f}")
            print("=" * 80)

        elif cmd == "verify":
            pipeline = ShippingCostPipeline()
            print("Running 3 consecutive passes to verify deterministic reproducibility...")
            passed = pipeline.verify_reproducibility(runs=3)
            if passed:
                print("Verification SUCCESS: All 3 runs produced 100% identical numbers, flags, and notes.")
            else:
                print("Verification FAILED: Output varied across runs.")
                sys.exit(1)

        elif cmd == "eval":
            harness = EvaluationHarness()
            results = harness.run_evaluation()
            print(f"Evaluation benchmark complete.")
            print(f"Total test cases: {results['total_cases']}")
            print(f"Accuracy: {results['accuracy_pct']:.1f}%")
            print(f"Hallucination rate: {results['hallucination_rate_pct']:.1f}%")

        elif cmd == "ask":
            pipeline = ShippingCostPipeline()
            verdicts, _ = pipeline.run()
            qa = AssistantQAService(verdicts)
            answer = qa.answer_question(parsed.question)
            print("\nAssistant Answer:\n")
            print(answer)


if __name__ == "__main__":
    cli = ShippingCostCLI()
    cli.execute()
