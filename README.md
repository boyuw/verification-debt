# verification-debt

Code, logs and figures for *Verification Debt: The Hidden Balance Sheet of Agent-Produced Work*.

The repository is deliberately small. Four synthetic task families with planted ground truth, a
local-only model backend, a producer/verifier/rework harness, a deterministic defect census that
never sees a prompt, the accounting layer that turns logs into `S`, `V`, `R` and `D`, and the
scripts that produce every number and figure in the paper.

```
verification-debt/
├── backends/ollama_backend.py   # local Ollama only; refuses non-loopback hosts
├── tasks/
│   ├── base.py                  # TaskFamily contract, Instance, Defect, JSON extraction
│   ├── invoices.py              # three-way match, planted price/qty/tax/duplicate/receipt defects
│   ├── emails.py                # policy KB + inbox, planted edge cases, action set + reply
│   ├── cleansing.py             # messy records -> entities, planted contradiction + merge trap
│   └── reports.py               # two operational tables, planted trend / anomaly / AHT drift
├── agents/
│   ├── producer.py              # baseline and guarded production prompts
│   ├── verifier.py              # independent verification pass (and the dual pass)
│   └── rework.py                # repair loop gated by the verifier, with effort accounting
├── measure/
│   ├── census.py                # defect census: deterministic grading vs planted ground truth
│   ├── effort.py                # S / V / R in one unit; char-per-token calibration
│   └── debt.py                  # ratios, bootstrap, rank correlation, guardrail deltas, leakage
├── run_all.py                   # pilot | sweep | summarize
├── make_figs.py                 # results/summary.json -> figures/*.pdf
├── scripts/selftest.py          # the census must accept gold and reject corruption
├── scripts/probe.py             # development probe (one instance per family)
├── results/                     # raw JSONL, summary.json, families.csv, figures
└── CLAIMS.md                    # every paper claim -> the results file or identity behind it
```

## Requirements

* Python 3.11+ with `numpy`, `scipy`, `matplotlib`
* [Ollama](https://ollama.com) on `http://localhost:11434`
* `ollama pull qwen2.5:14b-instruct` and `ollama pull llama3.1:8b`

No API keys, no paid endpoints. `backends/ollama_backend.py:assert_local` raises on any non-loopback
host, so the released code cannot accidentally call a hosted service.

---

# 1. Pre-registration

**Written before any debt experiment was run.** The only execution before this section existed was a
single-instance plumbing check per family (one work product, used to confirm the harness parses JSON
and the census scores it); no debt ratio was computed from it and no parameter below was chosen from
its outcome. The instance generators, the rubrics, the severity rules and the bars below were fixed
at the time of writing. Any change after this point is recorded in §1.7.

## 1.1 The quantity

For a task family `f` and a deployment configuration:

| symbol | meaning | measured as |
|---|---|---|
| `S` | gross production saving: human-equivalent production cost avoided | tokens to read the raw inputs + tokens to write the full correct deliverable |
| `V` | verification cost: effort to check the product to a fixed acceptance standard | tokens processed by the independent verifier, summed over every pass |
| `R` | rework cost: effort to bring failed products to acceptance | tokens processed by every repair call |
| `D` | debt ratio | `(V + R) / S` |

Accounting identity: booked ROI counts `S`; true net gain is `S − (V + R) = S(1 − D)`.

**`S` baseline procedure (fixed here, before any run).** `S` is measured against a *canonical
deliverable*: the complete, correct product serialised in the family's output schema, including a
one-line justification per field, generated deterministically by the family's own generator. The
saving is priced as

```
S = len(raw inputs) / rho_in  +  len(canonical deliverable) / rho_out
```

where `rho_in`, `rho_out` are characters-per-token ratios calibrated on the run's own traffic
(`measure/effort.py:calibrate`). `S` therefore prices the volume of work the artifact contains, not
the clock time of the person who would otherwise have done it. This is a volume proxy and is
discussed as such in the paper. `S` is a property of the instance, not of the arm or the model, so
the same denominator is used for baseline and guardrail runs.

`S` is matched across families by construction (instance sizes were chosen so that mean `S` falls in
a narrow band; see `results/summary.json:calibration`). This matters: `D` has `S` in the
denominator, so unmatched `S` would produce differences in `D` that have nothing to do with
verification.

## 1.2 Pre-registered success bars

* **B1.** At least one family shows `D ≥ 1.0` under the baseline configuration (pooled over both
  models).
* **B2.** `D` orders monotonically with the independently rated verifiability score across the four
  families: Spearman(verifiability, `D`) `≤ −0.8` with an exact permutation `p ≤ 0.10`. A single
  inversion is reported and discussed, not hidden.
* **B3.** Guardrails cut `D` by `≥ 30%` on the worst baseline family at `≤ 15%` throughput cost,
  where throughput cost is the percentage increase in total effort (production + verification +
  rework) per work product.

A missed bar is reported as a finding with its measured value. Rubrics, caps, baselines and severity
rules are not changed after seeing results. Any post-hoc variant is labelled EXPLORATORY.

## 1.3 Verifiability rubric (scored before any run, blind to results)

Three sub-scores, each 0–3, summed and divided by 9:

* **G — ground-truth checkability.** 3 = a single deterministic comparison per field against a
  planted value; 1 = the check requires re-deriving statistics from a table; 0 = no objective check.
* **R — rule determinism.** 3 = the acceptance rule is fully specified in the prompt; 2 = the rule is
  specified but its application requires a normalisation or clustering decision; 1 = the rule is
  stated in prose that the checker must interpret; 0 = the standard is a matter of judgement.
* **J — judgement required (reverse-scored).** 3 = no judgement beyond arithmetic; 2 = one local
  judgement per item; 1 = the checker must read free prose against a rule; 0 = the checker must
  decide what the deliverable should have said.

| family | G | R | J | score | why |
|---|---|---|---|---|---|
| invoices | 3 | 3 | 3 | **1.000** | every field is a planted number; the rule is arithmetic; no prose |
| cleansing | 3 | 2 | 2 | **0.778** | fields are checkable, but clustering into entities needs a local judgement |
| emails | 2 | 2 | 1 | **0.556** | the action set is checkable; the reply must be read against the policy |
| reports | 1 | 1 | 0 | **0.222** | the claim set is chosen by the producer; every claim must be re-derived; prose can be wrong and still read well |

Ordering hypothesis, stated up front: **`D` increases as verifiability decreases**, because `V` rises
(checking is nearly as hard as doing) and `R` rises (defects are subtler and survive first-pass
checks).

A second rater scores the same four families from the same rubric using only the family
specifications, never the results; both scores and their agreement are reported.

## 1.4 Measurement protocol

1. **Planted ground truth.** Each instance is generated together with its correct deliverable. The
   generator is the only source of truth; nothing is inferred from model output.
2. **Production.** The producer sees the raw inputs and the task rules, never the ground truth and
   never the verifier's rubric.
3. **Independent verification.** A separate pass, different system prompt, no access to the
   producer's reasoning, mandated to re-derive the answer before comparing. The verifier is the same
   model that produced the work (self-verification): each model is evaluated as a complete agent that
   both produces and checks its own output. This keeps the experimental surface small and the run
   tractable on a single GPU; verifier reliability is measured and reported as part of the analysis
   (see §1.7, D-VERIF). In the guarded arm the pass is run twice and both must accept.
4. **Rework.** A rejected product is returned with the reviewer's findings and repaired, then
   re-verified. Loop until acceptance or **5 rounds**, whichever comes first.
5. **Census.** The deterministic grader scores every product against planted truth, at first pass and
   at the end, with defects counted by severity (critical / major / minor) under the fixed rules in
   each family module.
6. **Acceptance standard for the census:** no critical and no major defect. The verifier's own
   acceptance is recorded separately; the difference between the two is reported as *leakage*.

Severity rules are fixed per family and documented in the module docstrings; in every family a
"false accept" (approving defective work) is critical and a "false reject" is major, so that the
census prices the error the organization actually pays for.

## 1.5 Configurations

| item | value |
|---|---|
| models | `qwen2.5:14b-instruct` (primary), `llama3.1:8b` (secondary); each produces **and** verifies its own work |
| decoding | temperature 0.0, top_p 1.0, seed 13, `num_ctx` 4096, `num_predict` 1536 (producer + rework) / 900 (verifier), `repeat_penalty` 1.05 |
| baseline arm | 35 work products per family per model |
| guardrail arm | 25 work products per family per model |
| rework cap | 5 rounds |
| instance seed | 13 |

The guardrail arm adds, on all four families: a structured acceptance checklist the producer must
complete (recorded in the `checklist` object) and a mandated dual verification pass before delivery.
Nothing else changes.

## 1.6 Analysis

Primary estimate of `D_f` is the ratio of sums over the family's instances, with a 4000-draw
instance bootstrap for the interval. The mean of per-product ratios is reported alongside. The rank
correlation uses the pre-registered verifiability score, with an exact permutation p-value (4! = 24
permutations). Guardrail deltas are computed on the same instances for both arms.

## 1.7 Deviations log

*Append-only. Every change to a pre-registered element is recorded here with its reason.*

| date | change | reason | effect on results |
|---|---|---|---|
| — | — | — | — |
| 2026-10-28 | **D-VERIF.** Verification is performed by the same model that produced the work (self-verification), reverting an earlier plan to use one fixed competent model for all producers. | The fixed-verifier plan tripled the run time (the competent model would have to verify both producers on a single GPU) for no validity gain once the verdict-parsing bug was fixed: verifier reliability is measured and reported regardless, and self-verification is the smaller, more honest experimental surface. | Each model is evaluated as a complete agent (produce + self-check). `D` therefore reflects the full cost of deploying that model, including its own (possibly weaker) verification. |
| 2026-10-28 | **D-VERIF-BUDGET.** Verifier `num_predict` raised 200→900 and the `accepted` field moved to the front of the verifier JSON schema in all four families. | The verifier's defect list exceeded the old 200-token budget and truncated mid-object, so `extract_json` returned nothing and every product was silently rejected, inflating `R` and making B1 true for the wrong reason. | Verifier verdicts now parse reliably; `accepted_by_verifier` reflects real rework need. |
| 2026-10-28 | **D-RP.** `repeat_penalty` 1.05 added to the Ollama decoding options, plus a salvage parser for truncated verifier verdicts. | The 8B model occasionally enters a repetition loop while emitting its verdict JSON and never closes the object, so the verdict could not be parsed and the product was silently escalated. The salvage parser recovers `accepted` and every complete defect object from a truncated tail. | Verifier verdicts parse reliably: 0 parse failures across 21 probe instances on all four families, both models. |
| 2026-10-28 | **D-N.** Sample size reduced from the pre-registered 150 per family per model (baseline) and 60 (guardrail) to **35 / 25**. | Measured single-GPU throughput is ~37 s wall clock per work product with 3 workers (the 14B model serialises all inference); 150 per cell would have required ~17 h of continuous compute. Protocol, bars and rubric are unchanged. | `D` is estimated over 35 (baseline) / 25 (guardrail) products per family per model; bootstrap intervals are wider than the pre-registered design intended and are reported as measured. |
| 2026-10-29 | **D-REVIEW.** Added a positive control (gold deliverable through verifier + census, `scripts/gold_control.py`), recomputed the guardrail comparison on matched `(model, iid)` pairs with a paired bootstrap CI, and corrected the §5 `S` unit from a family sum to a per-product mean. | External review found that (i) B1 is guaranteed by the unit asymmetry, which the control confirms (`D_gold = 1.44–2.25`), (ii) the guardrail table had compared against the full baseline rather than the matched sample, and (iii) §5 mislabelled the family sum of `S` as a mean. | B1 is now reported as *met but structural*; B3's ΔD is on matched instances with a CI that excludes zero; `S` is quoted per product. No protocol element, bar, or rubric changed; the control is the only new model run. |
| 2026-10-30 | **D-REVIEW2 (blocking).** Producer (and rework) `num_predict` raised 340→1536 in `agents/producer.py`, `agents/rework.py`, and the recorded config in `run_all.py`. | Round-2 review hypothesised that the 340-token producer cap could truncate outputs and manufacture spurious critical defects. Verified empirically: with the old 340 cap, `product_first` was truncated mid-JSON (failing to parse, ending without `}`) for **31/70 emails (44%)** and **50/70 cleansing (71%)** baseline products (qwen); the census then flagged a critical defect on every such product, so the 0/480 first-pass acceptance rate was partly an artifact. The guardrail arm's mandatory checklist shares the same 340 budget, so it was truncated too. A single-sample check with 1536 confirmed all four families now emit complete, parseable JSON (cleansing 387, emails 363, invoices 193, reports 257 completion tokens). | All 480 baseline + guardrail records were re-generated from scratch (old `results/raw/*.jsonl` archived to `results/raw_trunc340_archive/`). The re-run removes the truncation artifact: producer outputs are now complete, so first-pass acceptance and `R` reflect real defects, not cut-off JSON. `D` levels, B1/B3 and the verifier-reliability numbers in the paper are recomputed from the re-run and must be re-read against the new logs; the unit asymmetry and the positive-control finding are unchanged. |

---

# 2. Reproducing the paper

```bash
# 0. sanity: the census must accept gold and reject corruption (no model involved)
python scripts/selftest.py

# 1. pilot: one product per family, prints effort and acceptance
python run_all.py pilot

# 2. the pre-registered runs (~hours on one workstation; resumable, safe to interrupt)
python run_all.py sweep --n 35 --n-guard 25 --workers 3

# 3. aggregate -> results/summary.json, results/families.csv
python run_all.py summarize

# 4. figures -> figures/*.pdf
python make_figs.py

# 5. paper (numbers.tex and figures/*.pdf must exist first)
tectonic paper.tex
```

`run_all.py sweep` appends one JSONL record per work product to `results/raw/<arm>__<model>.jsonl`
and skips `run_id`s already present, so an interrupted sweep resumes where it stopped.

## 3. What is in a record

```json
{"run_id":"invoices|qwen2.5:14b-instruct|baseline|invoices-0001",
 "family":"invoices","iid":"invoices-0001","model":"...","arm":"baseline","seed":13,
 "inst_chars":1624,"canon_chars":1363,"n_checks":18,
 "prod_tokens":1220,"ver_tokens":1380,"rework_tokens":0,
 "prompt_tokens":... ,"completion_tokens":...,"in_chars":...,"out_chars":...,
 "n_rework_rounds":0,"accepted_by_verifier":true,
 "census_first":{"counts":{"critical":0,"major":2,"minor":1},"...":""},
 "census_final":{...},"product_first":"...","product_final":"...",
 "rounds":[{"round":0,"kind":"produce","verifier":{...}}],
 "config":{"temperature":0.0,"seed":13,...},"ts":"..."}
```

Every record carries the decoding configuration, the seed, and the token and wall-clock accounting
for production, verification and rework separately, so any number in the paper can be recomputed
from the logs alone.

## 4. Unit of effort, stated plainly

`V` and `R` are **model tokens processed**, measured by the local server. They are a proxy for human
verification effort, not a measurement of it. The paper devotes a subsection to the direction of the
bias and reports a sensitivity band over human/agent conversion factors. No number in the paper
converts agent time into human hours or dollars without stating the factor used.
