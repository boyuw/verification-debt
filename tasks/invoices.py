"""Family 1 - invoice exception handling.

Six three-way-match documents per instance.  One document per instance carries a
planted duplicate; the rest carry at most one planted discrepancy each, drawn
from {price, quantity, tax, missing receipt}.  The correct disposition is a pure
function of the documents, so the census needs no judgement at all.

Disposition rules (also stated to the producer and to the verifier):

  approve   invoice qty == received qty == PO qty, invoice unit price == PO unit
            price, invoice tax rate == PO tax rate, receipt present, not a
            duplicate of an earlier document in the batch.
  hold      anything else.
  escalate  hold AND invoice_total > 5000.

  payable   approve            -> invoice_total
            duplicate          -> 0.00
            missing receipt    -> 0.00
            otherwise          -> received_qty * po_unit_price
                                  (tax is checked for agreement but does not enter the payable)

  reason    priority: duplicate > no_receipt > qty > price > tax > ok
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import Defect, Instance, TaskFamily, as_float

ITEMS = ["A-118", "A-204", "B-330", "B-415", "C-072", "C-501", "D-190", "D-260"]
VENDORS = ["Northwind Supply", "Cobalt Industrial", "Meridian Parts", "Halcyon Metals",
           "Vector Components", "Ardent Packaging"]
TAX_RATES = [0.05, 0.08, 0.10]
DEFECT_KINDS = ["price", "qty", "tax", "no_receipt", "duplicate", "none"]


class InvoiceFamily(TaskFamily):
    name = "invoices"
    label = "Invoice exception handling"
    vis = {"ground_truth_checkability": 3, "rule_determinism": 3, "judgement": 3}
    rubric = (
        "For each document, re-derive the disposition from the three-way match and check "
        "three fields.\n"
        "  (1) decision: approve if quantity, unit price and tax rate agree across PO, "
        "receipt and invoice, the receipt exists, and the document is not a repeat of an "
        "earlier document id in the batch; otherwise hold; escalate instead of hold when "
        "invoice_total > 5000.\n"
        "  (2) payable: approve -> invoice_total; duplicate or missing receipt -> 0.00; "
        "otherwise received_qty * po_unit_price, rounded to 2 decimals (tax is checked for "
        "agreement but does not enter the payable).\n"
        "  (3) reason: the highest-priority discrepancy, ordering duplicate > no_receipt > "
        "qty > price > tax > ok.\n"
        "A product is accepted only if all three fields are correct for all six documents."
    )
    checklist = [
        "Every document id in the batch appears exactly once in your output.",
        "For each document, quantity, unit price and tax rate were compared across PO, receipt and invoice.",
        "Any repeated document id was marked duplicate and given payable 0.00.",
        "Every missing receipt was given payable 0.00.",
        "Every payable value is received_qty * PO unit price, not copied from invoice_total.",
        "Any hold with invoice_total above 5000 was escalated rather than held.",
    ]

    # ------------------------------------------------------------------ generate
    def generate(self, n: int, seed: int = 0) -> List[Instance]:
        rng = random.Random(seed * 7919 + 11)
        out: List[Instance] = []
        for i in range(n):
            out.append(self._one(rng, i))
        return out

    def _one(self, rng: random.Random, idx: int) -> Instance:
        n_doc = 6
        kinds = ["none"] + DEFECT_KINDS[:-1]           # one of each, then shuffle
        rng.shuffle(kinds)
        docs: List[Dict[str, Any]] = []
        ids: List[str] = []
        for k in range(n_doc):
            kind = kinds[k % len(kinds)]
            if kind == "duplicate" and k < 2:
                kind = "none"                          # a duplicate needs an earlier id
            doc_id = f"INV-{rng.randint(1000, 9899)}"
            if kind == "duplicate":
                doc_id = ids[rng.randrange(len(ids))]
            ids.append(doc_id)

            item = rng.choice(ITEMS)
            po_qty = rng.randint(6, 40)
            po_price = round(rng.uniform(12.0, 240.0), 2)
            tax = rng.choice(TAX_RATES)

            recv_qty = po_qty
            inv_qty, inv_price, inv_tax = po_qty, po_price, tax
            if kind == "qty":
                inv_qty = po_qty + rng.choice([-3, -2, -1, 2, 4])
                recv_qty = po_qty
            elif kind == "price":
                inv_price = round(po_price * rng.choice([0.94, 0.97, 1.03, 1.06]), 2)
            elif kind == "tax":
                inv_tax = 0.0 if tax != 0.0 else 0.08
            elif kind == "no_receipt":
                recv_qty = None

            inv_total = round(inv_qty * inv_price * (1 + inv_tax), 2)
            docs.append({
                "id": doc_id, "kind": kind, "item": item, "vendor": rng.choice(VENDORS),
                "po_qty": po_qty, "po_price": po_price, "po_tax": tax,
                "recv_qty": recv_qty, "inv_qty": inv_qty, "inv_price": inv_price,
                "inv_tax": inv_tax, "inv_total": inv_total,
                "day": rng.randint(1, 27),
            })

        truth: List[Dict[str, Any]] = []
        seen: List[str] = []
        for pos, d in enumerate(docs, 1):
            reasons: List[str] = []
            if d["id"] in seen:
                reasons.append("duplicate")
            seen.append(d["id"])
            if d["recv_qty"] is None:
                reasons.append("no_receipt")
            if d["kind"] == "qty" or (d["recv_qty"] is not None and d["inv_qty"] != d["recv_qty"]):
                reasons.append("qty")
            if abs(d["inv_price"] - d["po_price"]) > 1e-9:
                reasons.append("price")
            if abs(d["inv_tax"] - d["po_tax"]) > 1e-9:
                reasons.append("tax")

            order = ["duplicate", "no_receipt", "qty", "price", "tax"]
            reason = next((r for r in order if r in reasons), "ok")
            if reason == "ok":
                decision, payable = "approve", round(d["inv_total"], 2)
            else:
                if reason == "duplicate" or reason == "no_receipt":
                    payable = 0.0
                else:
                    payable = round(d["recv_qty"] * d["po_price"], 2)
                decision = "escalate" if d["inv_total"] > 5000 else "hold"
            truth.append({"doc": pos, "invoice_id": d["id"], "decision": decision,
                          "payable": payable, "reason": reason})

        text = self._render(docs)
        canonical = self._canonical(docs, truth)
        return Instance(
            family=self.name, iid=f"{self.name}-{idx:04d}", text=text, truth=truth,
            n_checks=3 * n_doc, canonical=canonical,
            meta={"n_docs": n_doc, "kinds": [d["kind"] for d in docs]},
        )

    def _render(self, docs: List[Dict[str, Any]]) -> str:
        lines = ["PURCHASE-TO-PAY BATCH", "Documents are listed in processing order.", ""]
        for d in docs:
            lines.append(f"[{d['id']}]  vendor: {d['vendor']}   received: 2026-03-{d['day']:02d}")
            lines.append(f"  PO line      item {d['item']}  qty {d['po_qty']}  "
                         f"unit_price {d['po_price']:.2f}  tax_rate {d['po_tax']:.2f}")
            if d["recv_qty"] is None:
                lines.append("  Receipt      NO RECEIPT ON FILE")
            else:
                lines.append(f"  Receipt      item {d['item']}  received_qty {d['recv_qty']}")
            lines.append(f"  Invoice      item {d['item']}  qty {d['inv_qty']}  "
                         f"unit_price {d['inv_price']:.2f}  tax_rate {d['inv_tax']:.2f}  "
                         f"invoice_total {d['inv_total']:.2f}")
            lines.append("")
        return "\n".join(lines)

    def _canonical(self, docs: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> str:
        note = {
            "ok": "three-way match agrees on quantity, unit price and tax rate; pay as invoiced",
            "price": "invoiced unit price differs from the purchase order; pay at the PO price",
            "qty": "invoiced quantity differs from the received quantity; pay for what arrived",
            "tax": "invoiced tax rate differs from the purchase order; pay at the PO tax rate",
            "duplicate": "document repeats an id already processed in this batch; pay nothing",
            "no_receipt": "no goods receipt on file, so the liability cannot be confirmed",
        }
        items = []
        for d, t in zip(docs, truth):
            items.append({
                "doc": t["doc"], "invoice_id": d["id"], "decision": t["decision"],
                "payable": t["payable"], "reason": t["reason"],
                "note": note[t["reason"]] + f" (invoice_total was {d['inv_total']:.2f})",
            })
        return '{"dispositions": ' + __import__("json").dumps(items, indent=1) + "}"

    # ------------------------------------------------------------------ prompts
    def producer_system(self) -> str:
        return (
            "You are an accounts-payable clerk processing an invoice batch. For each document, "
            "perform a three-way match across the purchase order, the goods receipt and the "
            "invoice, then decide the disposition.\n\n"
            "RULES\n"
            "  approve   quantity, unit price and tax rate agree across PO, receipt and invoice; "
            "a receipt exists; the document id has not already appeared earlier in the batch.\n"
            "  hold      anything else.\n"
            "  escalate  a hold whose invoice_total exceeds 5000.\n"
            "  payable   approve -> invoice_total. Duplicate or missing receipt -> 0.00. "
            "Otherwise received_qty * PO unit_price, rounded to 2 decimals. Tax is checked "
            "for agreement but is not part of the payable.\n"
            "  reason    the highest-priority discrepancy, ordering "
            "duplicate > no_receipt > qty > price > tax > ok.\n\n"
            "Respond with JSON only:\n"
            '{"dispositions":[{"doc":1,"invoice_id":"INV-1234","decision":"approve|hold|escalate",'
            '"payable":0.00,"reason":"ok|price|qty|tax|duplicate|no_receipt"}]}\n'
            "One entry per document. doc is the 1-based position of the document in the batch; "
            "two documents can share an invoice_id when one repeats the other, so doc is what "
            "identifies a row."
        )

    def verifier_system(self) -> str:
        return (
            "You are an independent accounts-payable reviewer. You did not produce the product "
            "under review and you must not assume it is correct. Recompute the correct "
            "disposition for every document yourself from the raw inputs, then compare.\n\n"
            "Respond with JSON only:\n"
            '{"accepted":true|false,"checks":<int>,"defects":[{"invoice_id":"...","field":"decision|payable|reason",'
            '"severity":"critical|major|minor","expected":"<your recomputed value>"}]}\n'
            "Set accepted to false if any document has an incorrect decision or an incorrect "
            "payable amount. An incorrect reason code alone does not block acceptance."
        )

    # ------------------------------------------------------------------ grading
    def census(self, inst: Instance, product: Optional[dict]) -> List[Defect]:
        if not product:
            return [Defect("unparseable_product", "critical", "no JSON product returned")]
        rows = product.get("dispositions")
        if not isinstance(rows, list):
            return [Defect("missing_dispositions", "critical", "no dispositions array")]
        got: Dict[int, dict] = {}
        for pos, r in enumerate(rows, 1):
            if not isinstance(r, dict):
                continue
            key = r.get("doc")
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = pos
            got[key] = r

        defects: List[Defect] = []
        for t in inst.truth:
            inv_id = t["invoice_id"]
            if t["doc"] not in got:
                defects.append(Defect("missing_document", "critical", f"doc {t['doc']} {inv_id}"))
                continue
            g = got[t["doc"]]
            dec = str(g.get("decision", "")).strip().lower()
            if dec != t["decision"]:
                if t["decision"] in ("hold", "escalate") and dec == "approve":
                    defects.append(Defect("false_accept", "critical",
                                          f"{inv_id}: approved a document that should be {t['decision']}"))
                elif t["decision"] == "approve":
                    defects.append(Defect("false_reject", "major",
                                          f"{inv_id}: {dec or 'blank'} on a clean document"))
                else:
                    defects.append(Defect("wrong_disposition", "minor",
                                          f"{inv_id}: {dec} vs {t['decision']}"))
            pay = as_float(g.get("payable"))
            if pay is None:
                defects.append(Defect("payable_missing", "major", inv_id))
            else:
                diff = pay - t["payable"]
                if abs(diff) > 0.02:
                    if t["payable"] <= 0.02 and pay > 0.02:
                        sev = "critical"                      # paying money that is not owed
                    else:
                        over = t["payable"] > 0 and diff / max(t["payable"], 1e-9) > 0.05
                        sev = "critical" if over else "major"
                    defects.append(Defect("overpayment" if diff > 0 else "underpayment",
                                          sev, f"{inv_id}: {pay:.2f} vs {t['payable']:.2f}"))
            reason = str(g.get("reason", "")).strip().lower()
            if reason != t["reason"]:
                defects.append(Defect("wrong_reason", "minor", f"{inv_id}: {reason} vs {t['reason']}"))
        return defects
