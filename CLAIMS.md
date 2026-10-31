# CLAIMS.md

Every quantitative claim in `paper.tex`, mapped to the file (or accounting identity) that
produces it. No number in the paper is typed by hand: `emit_numbers.py` generates `numbers.tex`
from `results/summary.json`, and `summary.json` is a pure function of `results/raw/*.jsonl`
through `measure/effort.py` and `measure/debt.py`. If a claim cannot be traced to a file below,
it does not belong in the paper.

## Measured outcomes (read these first)

| bar | pre-registered | measured | verdict |
|---|---|---|---|
| **B1** | some family `D ≥ 1.0` (baseline) | worst family invoices `D = 8.89` [7.83, 9.97]; all four families 6.62–8.89 | **met, but structural** — the positive control gives `D_gold = 1.44–2.25` with one pass and no rework, so B1 is guaranteed by the unit asymmetry, not discovered |
| **B2** | `D` rises as verifiability falls: Spearman ≤ −0.8, p ≤ 0.10 | Spearman **+0.40**, p = 0.75 (primary +0.20, secondary 0.00) | **missed — no detectable gradient** (with n=4 the bar effectively required ρ = −1) |
| **B3** | guardrail cuts `D` ≥ 30% at ≤ 15% throughput cost | matched-instance `D` **rose** 72–112%; throughput cost 63–101%; paired ΔD CI excludes 0 on all four families | **missed — direction reversed** |

These are reported in the paper as findings, not tuned away. The pre-registration (§1.2 of the
README) explicitly says a missed bar is reported with its measured value. Nothing in the protocol,
rubric, severity rules, or bars was changed after seeing results.

**Positive control (added after review).** The planted gold deliverable, fed through the same
verifier and census, both models (5 instances/family × 4 families × 2 models = 40): the census
accepts every gold product, the verifier accepts **1/40**, and `D_gold` is 1.44–2.25 (false-reject
39/40, 95% CI 87–100%). This is the single most important number in the paper because it shows the
headline ratio is dominated by a structural floor plus verifier false rejection, not by the work.
Source: `results/gold_control__*.json`, produced by `scripts/gold_control.py`.

| # | Claim (paper location) | Source | Note |
|---|---|---|---|
| C1 | Total work products `\VDnumRecords`; baseline arm `\VDnBaseline`; per-cell n `\VDnBase`/`\VDnGuard` (abstract, §6 Scale) | `results/summary.json: n_records, n_per_cell` (baseline total = 4 families × 70) | Reduced from the pre-registered 150/60; see README §1.7 D-N. First-pass counts (e.g. 22) are out of `\VDnBaseline` (280), not `\VDnumRecords` (480, which includes the guardrail arm). |
| C2 | `S` matched across families: per-product range `\VDsmin`–`\VDsmax`, ratio `\VDsratio` (§5) | `summary.json: pooled[].S` divided by `n` | `S = inst_chars/rho_in + canon_chars/rho_out` (`measure/effort.py:saving_tokens`); Table 1 shows the family **sum**, §5 quotes the **per-product** figure. |
| C3 | Chars-per-token calibration `\VDrhoIn`, `\VDrhoOut` (§5) | `summary.json: calibration` | `measure/effort.py:calibrate`, estimated on the run's own traffic. |
| C4 | Per-family decomposition `S, V, R`, `V/S`, `R/S` (Table 2, Fig. 1) | `summary.json: pooled[]` (S/V/R in thousands via `\VD<Family>S/V/R`) | Ratio-of-sums over all baseline records of both models; S/V/R are family sums. |
| C5 | Debt ratios `D` with 95% bootstrap intervals, per family (Table 2, abstract) | `summary.json: pooled[].D, D_lo, D_hi` | `measure/debt.py:bootstrap_ratio`, 4000 draws; D = (V+R)/S by construction. |
| C6 | **B1**: at least one family `D ≥ 1.0`; worst family and its `D` (§7.1) | `summary.json: bars.B1_*`; `gold_control__*.json` for the floor | Computed in `run_all.py:summarize`; reported as met **but structural** (C17). |
| C7 | **B2**: Spearman `\VDspearman`, exact permutation `p=\VDpperm` (§7.2, Fig. 2) | `summary.json: pooled_rank`; verifiability scores in README §1.3 | `measure/debt.py:spearman/permutation_p` (4! = 24 orderings). With n=4 the bar required ρ = −1; stated in §7.2. |
| C8 | **B3**: guardrail `\Delta D` on the **same instances**, paired bootstrap CI (§7.3, Fig. 3) | `summary.json: pooled_guardrail[]` (`D_base` matched, `delta_lo/hi`) | `measure/debt.py:guardrail_table` now restricts baseline to the (model, iid) pairs present in the guardrail arm. |
| C9 | First-pass vs final census acceptance per family (Table 2) | `summary.json: pooled[].pass1_accept, final_accept` | Deterministic census (`measure/census.py`), not a model judgement. With the 1536-token producer budget the census accepts cleansing only (`\VDCleansePass` first pass, `\VDCleanseFinal` final); the other three families stay at 0. |
| C10 | Verifier acceptance and rework rate per family (Table 2) | `summary.json: pooled[].verifier_accept, rework_rate, mean_rounds` | From the rework loop (`agents/rework.py`), gated by the verifier. |
| C11 | Leakage: `\VDqLeak` of verifier-accepted products still carry a census-blocking defect (§7.4) | `summary.json: leakage` | `measure/debt.py:leakage`; **n = 2**, so reported as a count, not a rate (A7). |
| C12 | Sensitivity band `D(λ)`, λ ∈ {0.5,1,2,4}: level scales, ordering λ-invariant (§5, Fig. 4) | `summary.json: sensitivity` | `D(λ) = λ(V+R)/S` linear in λ; `S` does not scale with λ. |
| C13 | Per-model `D` and per-model rank correlations (§7.5) | `summary.json: by_model, by_model.__rank` | Secondary analysis; ordering is model-dependent (stated). |
| C14 | The intervention raises `V` by construction and raises (not lowers) `R` (§7.3) | `summary.json: pooled_guardrail[]` | Dual-pass verification bills a second full pass; the verifier accepts essentially nothing in either arm (`\VDqBaseAcc` primary, `\VDlBaseAcc` secondary), so the acceptance channel is closed. |
| C15 | Verifier verdict parse failures ≈ 0 after the fixes (appendix `app:repeats`) | `scripts/diag_fail.py`; `rounds[].verifier.verdict_error` | Probe: 21 instances, all families, both models, 0 failures. |
| C16 | Positive control: `D_gold`, verifier false-reject on gold (Table 1, abstract, §5, §7.1) | `results/gold_control__qwen25-14b-instruct.json`, `...llama31-8b.json` | `scripts/gold_control.py`; 5 instances/family, both models (40 total); census accepts all gold, verifier accepts 1/40 (false-reject 39/40, 95% CI 87–100%). Fisher test is **matched caliber**: 1/40 gold vs 2/280 pooled baseline, `p = 0.33`. (A 1/40-vs-2/140 table mixes both-model gold with primary-only baseline and gives 0.53; that is a caliber error, not used.) |
| C17 | Break-even conversion `λ* = 1/D` (§5, §7.5) | `summary.json: pooled[].D` (macro `\VDlambdaStarMin/Max`) and `by_model[primary]` (`\VDlambdaStarQMin/Max`) | Identity. Pooled λ* = 0.11–0.15; primary-model λ* = 0.085–0.143 (the caliber Figure 4 uses). |
| C18 | Per-family check counts (§7.2) | `summary.json: n_checks` (macro `\VDchecks<Family>`) | 18/30/12/10; used to refute the "more checks ⇒ more debt" reading. |
| C19 | Loop termination is a measurement choice: `D` at cap ∈ {1,2,3,5}, patience 2 (worst family, pooled) | `summary.json: termination_sensitivity` (macros `\VDtermCapA/B/C/E`, `\VDtermPatA`) | `measure/debt.py:termination_sensitivity` replays stored per-round verdicts offline, pooled, as ratio-of-sums. `D` = 4.60 (cap 1) → 8.89 (cap 5 = Table 2). **The deployed rule reproduces the recorded pooled D exactly**, enforced by `tests/test_termination_replay.py`. |
| C20 | Prompt-token share of checker effort: `\VDpromptShareMin`–`\VDpromptShareMax` (§app:robust) | `summary.json: token_split` | `measure/effort.py`; the checker re-reads inputs/rubric/product, which has no analogue in `S`. |
| C21 | Verifier–census product-level agreement at first pass: precision `\VDdetPrec`, recall `\VDdetRec`, false-reject `\VDdetFalseRej` (§7.1) | `summary.json: detection_agreement` (macros `\VDdet*`, `\VDdetCleanse*`) | `measure/debt.py:detection_agreement`; 2×2 = TP 256 / FP 22 / FN 2 / TN 0 (positive = census flags a blocking defect). Error concentrated on cleansing (precision 69%; 31% of its rejections false) and gold (97.5%); the verifier accepted 0% of the 22 census-clean cleansing products. |
| C22 | Checker-attributable share of effort: `\VDchkShare` (95% CI `\VDchkShareLo`–`\VDchkShareHi`), `D_check = \VDchkD` (§7.1, abstract, discussion, conclusion) | `summary.json: checker_attribution` (macros `\VDchk*`) | `measure/debt.py:checker_attribution`; attributes the **whole V+R** of every trajectory whose *first* rejection was false (not just R). Share = 5.9% of all V+R (95% CI 3.5–8.7%); D_check = 0.45 (95% CI 0.27–0.65) vs total D 7.58. Supersedes the earlier R-only "6% of rework" and closes the inference gap (V was previously omitted). |

## Claims that are explicitly *not* made

* No claim converts agent tokens into human hours or dollars. The conversion band (λ) is
  reported as a sensitivity, with the factor stated (C12), and the break-even `λ* = 1/D` is given
  instead of a point conversion (C17).
* No claim that any real organisation's ROI was measured. All work products are synthetic; the
  construct is an accounting identity evaluated on them (§Limitations).
* No claim that the measured `D` is a property of agent work. The positive control (C16) shows it
  is dominated by a structural floor plus verifier false rejection; the paper says so in the
  abstract and §7.1.
* No claim of a verifiability gradient. B2 is reported as missed, with the power caveat (n=4,
  the bar required ρ = −1).
* The field-motivation section (§2) cites Brynjolfsson's productivity-paradox work and the
  companion paper (arXiv:2609.18126) as motivation only. Chen & Stratton is cited as motivation
  and is **not** used as evidence for the construct: the title, sample (718 firms, >700k workers)
  and effect directions were confirmed from public coverage, but the microdata are not public, so
  the numbers are quoted as reported rather than re-derived. This supersedes the earlier note that
  the manuscript was not retrievable; the title was corrected to *Artificial Intelligence in the
  Firm: Bottlenecks in Software Production* (August 2026 version).

## Post-review analyses (all on existing logs; no new generation except the control)

Added in response to an external review, and logged here so the audit trail is complete:

* **Positive control** (C16): `scripts/gold_control.py`, 5 instances/family × 2 models. This is
  the only new model run; it feeds the planted gold through verifier + census.
* **Matched guardrail** (C8): `guardrail_table` now matches on `(model, iid)` and reports a paired
  bootstrap CI. The previous version compared the guardrail against the full baseline sample, which
  contradicted the §6 claim that deltas were on the same instances. Fixed.
* **Unit/per-product fix** (C2): §5 previously called the family *sum* of `S` a "mean"; corrected.
* **λ\*** (C17), **check counts** (C18): added to address the unit-asymmetry and mechanism points.
* **Removed claims**: the wall-clock robustness claim (per-component time is not recoverable from
  the logs) and the `1/(1−D)` "overstatement factor" (undefined for `D ≥ 1`; replaced by
  `net = S(1−D) < 0`).

## Deviations that touch the numbers

Recorded in README §1.7 and summarised here because they affect how C1–C12 should be read:

* **D-N** — sample size 35/25 per family per model instead of the pre-registered 150/60, due to
  single-GPU throughput (~37 s per product). Intervals are wider than the design intended.
* **D-VERIF** — each model verifies its own output (self-verification), so `D` reflects the full
  cost of deploying that model, including its own verification quality.
* **D-VERIF-BUDGET / D-RP** — verifier token budget 900, `accepted` emitted first, repetition
  penalty 1.05, salvage parser for truncated verdicts. Without these the verifier's verdicts
  failed to parse and every product was silently rejected, which would have inflated `R`
  mechanically (C16).
* **D-PROD-BUDGET (re-run)** — the original 340-token producer/rework budget truncated many
  products mid-JSON. Raised to 1536 and the full 480-record run regenerated. The concern was that
  truncation manufactured a zero first-pass acceptance; it did the opposite. With room to finish the
  census now accepts cleansing at first pass (`\VDCleansePass`), yet the verifier still accepts
  essentially none of it (`\VDCleanseVerAcc`). So the near-zero *acceptance* is the verifier
  over-rejecting clean work, not a producer artefact — the same failure the gold control shows
  (C16). This is why C14/C16 and the "verifier, not the producer, is the bottleneck" reading in §7
  supersede the earlier "no product was accepted" framing.
