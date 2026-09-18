# Freight Tiger Shipping Cost Assistant

Autonomous shipping cost audit assistant that monitors freight records, detects anomalous route cost increases, retrieves real-world context notes via ChromaDB vector search, and generates grounded explanations using LangChain with strict anti-hallucination guardrails.

---

## 1. Project Structure

```
AI Intern Case Study/
|-- src/
|   |-- models.py
|   |-- data_loader.py
|   |-- metrics_calculator.py
|   |-- anomaly_detector.py
|   |-- rag_retriever.py
|   |-- evaluator.py
|   |-- pipeline.py
|   |-- eval_harness.py
|-- data/
|   |-- shipment_records.csv
|   |-- context_notes.csv
|   |-- sample_output_format_v2.csv
|-- output/
|   |-- output_submission.csv
|-- docs/
|   |-- FreightTiger_Intern_CaseStudy.pdf
|-- main.py
|-- README.md
```

---

## 2. Core Metrics & Math Parity

### Metric Definitions
- **Unit Cost**: $\text{Cost per tonne-km} = \frac{\sum \text{Freight Cost (INR)}}{\sum (\text{Quantity (Tonnes)} \times \text{Distance (km)})}$ grouped by calendar week (`week_of` = Monday date).
- **vs. own history**: Trailing 8-week rolling average strictly prior to the current week (no lookahead). If $< 8$ weeks exist, all available prior weeks are used without padding.
- **vs. similar routes**: Average unit cost in the same week across all other routes sharing the same `route_type` (`Short`/`Medium`/`Long`), strictly excluding the route itself.

### Verification Against Sample Outputs
| Route | Week | Metric | Expected | Computed | Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Delhi-Jaipur** | 2024-11-11 | Cost / vs own / vs peer | 4.17 / +35.5% / +21.0% | 4.17 / +35.5% / +21.0% | Exact |
| **Ahmedabad-Mumbai** | 2025-01-20 | Cost / vs own / vs peer | 3.29 / +29.5% / +22.5% | 3.29 / +29.5% / +22.5% | Exact |
| **Mumbai-Pune** | 2025-09-15 | Cost / vs own / vs peer | 3.98 / +9.2% / +23.6% | 3.98 / +9.2% / +23.6% | Exact |

---

## 3. Full Audit Results & Anomaly Breakdown

The pipeline analyzed **2,940 shipments** across **312 route-weeks** and detected **20 anomalous route-weeks** (cost surge >= 15% vs own 8-week history OR >= 20% vs peer routes).

### Audit Summary Statistics
- **Total Shipments Evaluated**: 2,940
- **Total Route-Weeks Evaluated**: 312
- **Anomalies Detected**: 20
- **Justified Surges (`flagged = No`)**: 3 (15.0%) - verified against real-world disruption events
- **Flagged for Review (`flagged = Yes`)**: 17 (85.0%) - unexplained spikes or non-rate-impacting trap notes

### Audited Submission Summary (`output_submission.csv`)

| Route | Week Of | Unit Cost | vs Own History | vs Peer Routes | Flagged | Note ID | Audit Verdict Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ahmedabad-Mumbai** | 2025-01-20 | 3.29 | +29.5% | +22.5% | **No (justified)** | N002 | Makar Sankranti festival driver crunch; temporary surcharge justified |
| **Chennai-Bangalore** | 2025-02-24 | 3.58 | +31.5% | +38.7% | **No (justified)** | N001 | Severe cyclone flooding forced detours and operational rerouting |
| **Chennai-Bangalore** | 2025-03-03 | 3.48 | +22.8% | +32.0% | **No (justified)** | N001 | Ongoing flood recovery and detour costs justified |
| **Chennai-Bangalore** | 2025-03-10 | 3.57 | +22.1% | +33.8% | **Yes** | - | Trap note N009: repairs completed; persistent spike unjustified |
| **Chennai-Bangalore** | 2025-03-17 | 3.47 | +13.7% | +33.5% | **Yes** | - | Route restored to normal (N009); peer premium remains unjustified |
| **Delhi-Jaipur** | 2024-11-11 | 4.17 | +35.5% | +21.0% | **Yes** | - | Unexplained surge (+35.5% own, +21.0% peer); zero external events |
| **Delhi-Jaipur** | 2024-11-18 | 4.09 | +27.6% | +19.7% | **Yes** | - | Follow-up elevated rate (+27.6% own); requires carrier billing audit |
| **Mumbai-Pune** | 2025-06-23 | 3.76 | +6.2% | +20.6% | **Yes** | - | Peer corridor premium (+20.6%); no operational disruption logged |
| **Mumbai-Pune** | 2025-09-15 | 3.98 | +9.2% | +23.6% | **Yes** | - | Trap note N006: market demand stable; cost surge unjustified |
| **Mumbai-Pune** | 2025-10-06 | 3.94 | +5.4% | +22.9% | **Yes** | - | Trap note N006: normal capacity; rate premium uncorroborated |
| **Mumbai-Pune** | 2025-10-20 | 4.10 | +7.7% | +24.8% | **Yes** | - | Trap note N010: GPS mandate absorbed without rate increase |
| **Mumbai-Pune** | 2025-10-27 | 4.07 | +5.5% | +28.3% | **Yes** | - | Mandate absorbed without rate hike (N010); flagged for review |
| **Mumbai-Pune** | 2025-11-03 | 4.20 | +7.4% | +24.1% | **Yes** | - | Mandate absorbed without rate hike (N010); flagged for review |
| **Mumbai-Pune** | 2025-11-10 | 3.97 | -0.1% | +21.4% | **Yes** | - | Peer group premium (+21.4%); no valid rate surcharge |
| **Mumbai-Pune** | 2025-11-17 | 4.32 | +8.9% | +31.6% | **Yes** | - | Severe peer premium (+31.6%); N010 absorbed without rate hike |
| **Mumbai-Pune** | 2025-11-24 | 4.16 | +3.0% | +23.0% | **Yes** | - | Persistent peer premium (+23.0%); flagged for investigation |
| **Mumbai-Pune** | 2025-12-01 | 4.39 | +7.7% | +37.4% | **Yes** | - | Major peer premium (+37.4%); uncorroborated carrier rate surge |
| **Mumbai-Pune** | 2025-12-08 | 4.43 | +7.2% | +38.8% | **Yes** | - | Extreme peer premium (+38.8%); no operational context logged |
| **Mumbai-Pune** | 2025-12-15 | 4.39 | +4.4% | +35.5% | **Yes** | - | Extended rate anomaly (+35.5% peer); carrier inquiry required |
| **Mumbai-Pune** | 2025-12-22 | 4.49 | +5.7% | +38.9% | **Yes** | - | Peak rate anomaly (+38.9% peer); flagged for freight reconciliation |

---

## 4. RAG Architecture & Anti-Hallucination Guardrails

### ChromaDB Vector Search + Temporal Matching
1. **Indexing**: 10 context notes are embedded into **ChromaDB** using `text-embedding-3-small`.
2. **Retrieval**: Candidate notes are retrieved via semantic vector similarity and filtered by corridor applicability (`applies_to == route` or `All Routes`) and active operational date windows.
3. **Trap Note Handling**: Notes stating costs were unaffected (e.g., N005), routes outside dataset (N004), or stable conditions (N006, N008, N010) are rejected as justifications.
4. **Verdict Policy**:
   - Valid justifying event confirmed: `flagged = "No (justified)"`, `matched_note_id = "<NoteID>"`.
   - Unexplained / non-justifying note: `flagged = "Yes"`, `matched_note_id = ""` (blank).

---

## 5. DeepEval Evaluation Benchmark

Evaluated in `src/eval_harness.py` using **DeepEval** (`HallucinationMetric`) and ground-truth validation:

| Scenario | Route | Week | Expected Flag | Note ID | DeepEval Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Festival Surcharge | Ahmedabad-Mumbai | 2025-01-20 | No (justified) | N002 | 1.00 (Faithful) | PASS |
| Flood Disruption | Chennai-Bangalore | 2025-02-24 | No (justified) | N001 | 1.00 (Faithful) | PASS |
| Unexplained Surge | Delhi-Jaipur | 2024-11-11 | Yes | (blank) | N/A | PASS |
| Demand Normal Note (N006) | Mumbai-Pune | 2025-09-15 | Yes | (blank) | N/A | PASS |
| Maintenance Trap Note (N005)| Mumbai-Delhi | 2024-07-29 | Yes | (blank) | N/A | PASS |
| Road Recovery Note (N009) | Chennai-Bangalore | 2025-03-17 | Yes | (blank) | N/A | PASS |

- **Benchmark Accuracy**: **100.0%** (6/6)
- **DeepEval Hallucination Score**: **1.00** (0.0% hallucination rate)

---

## 6. Reproducibility Check (3 Runs)

Verified across 3 untouched full runs (`uv run python main.py verify`):
- All numbers, flags, and cited note IDs are **100% identical** across passes (0 diffs).
- Determinism guaranteed via `temperature=0.0` and deterministic guardrail validation.

---

## 7. Token & Cost Log: Model Comparison

Full run over all ~2,940 shipment records (20 anomalous route-weeks audited):

| Model | Provider | Input Rate (per 1M) | Output Rate (per 1M) | Prompt Tokens | Completion Tokens | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt-4o-mini** | OpenAI | $0.15 | $0.60 | 1,948 | 1,998 | **$0.001491** |
| **gpt-5.6-luna** | OpenAI | $0.20 | $1.20 | 1,948 | 1,998 | **$0.002787** |

- **GPT-4o-mini** is the baseline production choice at **$0.00149 USD** (~0.12 INR) per complete 2-year audit.
- **GPT-5.6 Luna** costs **1.87x** due to higher output pricing ($1.20 / 1M).

---

## 8. Execution Commands

### Primary Commands
```bash
# Run full audit and export output_submission.csv
uv run python main.py run

# Run 3-pass reproducibility verification
uv run python main.py verify

# Run DeepEval benchmark evaluation harness
uv run python main.py eval

# Compare model inference pricing (GPT-4o-mini vs GPT-5.6 Luna)
uv run python main.py compare

# Plain-English Q&A about findings (Stretch Goal)
uv run python main.py ask "Why did Ahmedabad-Mumbai get pricier in January?"
```

### Standalone Module Tests
Each module in `src/` can be executed independently:
```bash
uv run python src/models.py
uv run python src/data_loader.py
uv run python src/metrics_calculator.py
uv run python src/anomaly_detector.py
uv run python src/rag_retriever.py
uv run python src/evaluator.py
uv run python src/pipeline.py
uv run python src/eval_harness.py
```
