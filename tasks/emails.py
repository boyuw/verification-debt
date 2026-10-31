"""Family 2 - customer-email resolution drafts.

Each instance ships a generated policy knowledge base (seven numbered rules with
randomised thresholds) and two customer emails whose facts are stated in prose,
never as flags.  The producer must draft a reply and record the account actions
it is taking.  Ground truth is the action set implied by the policy.

The family sits mid-range on verifiability: the *actions* are checkable, the
*prose* is only partly checkable, and the costliest defect (promising an outcome
the policy does not allow) is invisible unless the reviewer reads the draft
against the rules rather than against the action list.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from .base import Defect, Instance, TaskFamily, as_float

FIRST = ["Dana", "Priya", "Marcus", "Lena", "Tomas", "Aisha", "Ruth", "Kenji", "Sofia", "Ivan"]
LAST = ["Okafor", "Lindqvist", "Ramos", "Bergman", "Nakamura", "Duval", "Osei", "Petrov"]
PRODUCTS = ["the Halo desk lamp", "the Fjord travel mug", "the Atlas keyboard", "the Rill headphones",
            "the Tundra jacket", "the Sable office chair"]

FORBIDDEN = {
    "deny_refund": ["full refund", "have refunded", "refund has been issued", "will refund",
                    "refunded in full", "money back"],
    "opened_refund": ["full refund", "refunded in full", "waive the restocking fee",
                      "no restocking fee"],
    "defective_refund": ["restocking fee", "deduct", "partial refund"],
}


class EmailFamily(TaskFamily):
    name = "emails"
    label = "Customer-email resolution drafts"
    vis = {"ground_truth_checkability": 2, "rule_determinism": 2, "judgement": 1}
    rubric = (
        "For each email, derive the permitted action set from the policy rules and check four "
        "things.\n"
        "  (1) action types: the set of action types must equal the set the policy permits - "
        "no missing action, no extra action.\n"
        "  (2) amounts: any refund or discount amount must match the policy computation to "
        "within 0.50.\n"
        "  (3) citations: the reply must name the rule ids that govern the decision.\n"
        "  (4) promises: the reply must not promise an outcome the policy forbids (for example "
        "a full refund where a restocking fee applies, or a refund where the window has closed).\n"
        "A product is accepted only if (1), (2) and (4) hold for both emails."
    )
    checklist = [
        "Every fact you rely on (window, opened, defective, tier, order value, escalation) is quoted from the email or the policy, not assumed.",
        "The action set contains no action the policy does not permit.",
        "Every refund or discount amount is recomputed from the policy numbers, not estimated.",
        "The reply names the rule ids that govern the decision.",
        "The reply makes no promise that the action set does not deliver.",
        "Any escalation trigger (chargeback, legal threat, exception above the approval threshold) was honoured.",
    ]

    # ------------------------------------------------------------------ generate
    def generate(self, n: int, seed: int = 0) -> List[Instance]:
        rng = random.Random(seed * 6151 + 29)
        return [self._one(rng, i) for i in range(n)]

    def _one(self, rng: random.Random, idx: int) -> Instance:
        pol = {
            "N": rng.choice([21, 30, 30, 45]),
            "F": rng.choice([10, 15, 20, 25]),
            "D1": rng.choice([5, 8, 10]),
            "D2": rng.choice([12, 15, 18]),
            "X": rng.choice([250, 300, 400]),
            "ship": 6.95,
        }
        emails = []
        for k in range(3):
            scenario = ["defective", "opened_in_window", "late_window", "clean", "chargeback"][
                (idx * 3 + k) % 5
            ]
            a = {
                "email_id": f"TICKET-{rng.randint(10000, 99999)}",
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "product": rng.choice(PRODUCTS),
                "tier": rng.choice(["silver", "silver", "gold", "platinum"]),
                "order_value": round(rng.uniform(45, 620), 2),
                "days": rng.randint(2, 60),
                "opened": scenario in ("opened_in_window", "defective"),
                "defective": scenario == "defective",
                "chargeback": scenario == "chargeback",
                "scenario": scenario,
            }
            if scenario == "late_window":
                a["days"] = pol["N"] + rng.randint(2, 20)
            elif scenario in ("opened_in_window", "clean", "defective"):
                a["days"] = rng.randint(2, max(3, pol["N"] - 3))
            emails.append(a)

        truth: Dict[str, Any] = {}
        for a in emails:
            acts, cites = self._decide(a, pol)
            forb = []
            for act in acts:
                key = ("defective_refund" if a["defective"] else
                       "opened_refund" if (a["opened"] and act["type"] == "refund") else
                       act["type"])
                forb += FORBIDDEN.get(key, [])
            truth[a["email_id"]] = {"actions": acts, "cites": cites,
                                    "forbidden": sorted(set(forb))}

        text = self._render(pol, emails)
        canonical = self._canonical(emails, truth, pol)
        n_checks = 4 * len(emails)
        return Instance(family=self.name, iid=f"{self.name}-{idx:04d}", text=text,
                        truth=truth, n_checks=n_checks, canonical=canonical,
                        meta={"n_emails": len(emails), "policy": pol,
                              "scenarios": [a["scenario"] for a in emails]})

    @staticmethod
    def _decide(a: Dict[str, Any], pol: Dict[str, Any]) -> tuple:
        """Action set implied by the policy.

        R5 defines an *exception* as any resolution other than a standard
        in-window, unopened refund.  An exception above the value threshold is
        escalated instead of resolved, and the escalated amount is the money that
        would move if the supervisor approved it.
        """
        ov, ship = a["order_value"], pol["ship"]
        if a["chargeback"]:
            return [{"type": "escalate", "amount": 0.0}], ["R7"]
        if a["defective"]:
            amt = round(ov + ship, 2)
            if ov > pol["X"]:
                return [{"type": "escalate", "amount": amt}], ["R3", "R5"]
            return [{"type": "refund", "amount": amt}], ["R3"]
        if a["days"] > pol["N"]:
            if a["tier"] in ("gold", "platinum"):
                pct = pol["D1"] if a["tier"] == "gold" else pol["D2"]
                amt = round(ov * pct / 100.0, 2)
                if ov > pol["X"]:
                    return [{"type": "escalate", "amount": amt}], ["R1", "R4", "R5"]
                return ([{"type": "deny_refund", "amount": 0.0},
                         {"type": "discount", "amount": amt}], ["R1", "R4"])
            return [{"type": "deny_refund", "amount": 0.0}], ["R1"]
        if a["opened"]:
            amt = round(ov * (1 - pol["F"] / 100.0), 2)
            if ov > pol["X"]:
                return [{"type": "escalate", "amount": amt}], ["R2", "R5"]
            return [{"type": "refund", "amount": amt}], ["R2"]
        return [{"type": "refund", "amount": round(ov, 2)}], ["R1"]

    def _render(self, pol: Dict[str, Any], emails: List[Dict[str, Any]]) -> str:
        kb = (
            "CUSTOMER POLICY KNOWLEDGE BASE (current)\n"
            f"R1  Refunds are available for {pol['N']} calendar days from the delivery date.\n"
            f"R2  Returned items that have been opened are refunded less a {pol['F']}% restocking "
            "fee, calculated on the item price and excluding original shipping.\n"
            "R3  Items that arrive defective are refunded in full, including original shipping, "
            "with no restocking fee, at any time within the return window.\n"
            f"R4  Goodwill: gold members receive {pol['D1']}% and platinum members "
            f"{pol['D2']}% off a future order when a refund is declined.\n"
            f"R5  An exception is any resolution other than a standard in-window, unopened refund "
            f"(that is: a defective-item refund, a restocking-fee refund, or a goodwill discount). "
            f"Any exception on an order above {pol['X']:.2f} in value must be escalated to a "
            "supervisor rather than resolved; the escalated amount is the money that would move "
            "if the supervisor approved it.\n"
            "R6  Replacements ship within two business days; refunds post in five to seven "
            "business days.\n"
            "R7  Any mention of a chargeback, a card dispute or legal action must be escalated "
            "to the resolutions team immediately; do not offer or process a refund.\n"
            f"Standard outbound shipping on these orders was {pol['ship']:.2f}.\n"
        )
        body = []
        for a in emails:
            extra = []
            if a["defective"]:
                extra.append(f"It stopped working properly within a day of arriving "
                             f"({a['days']} days ago).")
            if a["opened"]:
                extra.append("I did open the box and try it out before deciding.")
            if a["chargeback"]:
                extra.append("If this is not sorted out this week I will file a chargeback "
                             "with my card issuer.")
            if a["days"] > pol["N"]:
                extra.append(f"It has been {a['days']} days since it arrived.")
            if a["tier"] != "silver":
                extra.append(f"I am a {a['tier']} member.")
            body.append(
                f"[{a['email_id']}]\nFrom: {a['name']}\nSubject: problem with my order\n\n"
                f"Hello, I bought {a['product']} for {a['order_value']:.2f} and it was "
                f"delivered {a['days']} days ago. " + " ".join(extra) +
                "\nPlease let me know what you can do.\n"
            )
        return kb + "\nINBOX\n------\n" + "\n".join(body)

    def _canonical(self, emails: List[Dict[str, Any]], truth: Dict[str, Any],
                   pol: Dict[str, Any]) -> str:
        drafts = []
        for a in emails:
            t = truth[a["email_id"]]
            act_txt = "; ".join(f"{x['type']} {x['amount']:.2f}" for x in t["actions"])
            drafts.append({
                "email_id": a["email_id"],
                "reply": (f"Thank you for writing about {a['product']}. Under "
                          f"{', '.join(t['cites'])} the permitted resolution is: {act_txt}. "
                          "Refunds post in five to seven business days (R6)."),
                "actions": t["actions"],
            })
        return json.dumps({"drafts": drafts}, indent=1)

    # ------------------------------------------------------------------ prompts
    def producer_system(self) -> str:
        return (
            "You are a customer-support agent. Read the policy knowledge base, then resolve each "
            "email in the inbox. Decide the account actions the policy permits and write the "
            "reply the customer will receive. The reply must be consistent with the actions: "
            "never promise something the policy does not allow.\n\n"
            "Respond with JSON only:\n"
            '{"drafts":[{"email_id":"TICKET-12345","reply":"<customer-facing text>",'
            '"actions":[{"type":"refund|discount|deny_refund|escalate|replace","amount":0.00}]}]}\n'
            "Use amount 0.00 for actions that move no money. One entry per email."
        )

    def verifier_system(self) -> str:
        return (
            "You are an independent quality reviewer for customer support. You did not write the "
            "draft under review. Derive the permitted action set yourself from the policy, then "
            "read the reply text against the policy - a reply can be wrong even when the action "
            "list is right, if it promises something the policy forbids.\n\n"
            "Respond with JSON only:\n"
            '{"accepted":true|false,"checks":<int>,"defects":[{"email_id":"...","field":"actions|amount|citation|promise",'
            '"severity":"critical|major|minor","expected":"<your value or the permitted action>"}]}\n'
            "Set accepted to false if any action is missing or impermissible, if any amount is "
            "wrong by more than 0.50, or if the reply promises a forbidden outcome."
        )

    # ------------------------------------------------------------------ grading
    def census(self, inst: Instance, product: Optional[dict]) -> List[Defect]:
        if not product:
            return [Defect("unparseable_product", "critical", "no JSON product returned")]
        rows = product.get("drafts")
        if not isinstance(rows, list):
            return [Defect("missing_drafts", "critical", "no drafts array")]
        got = {str(r.get("email_id", "")).strip(): r for r in rows if isinstance(r, dict)}

        defects: List[Defect] = []
        for eid, t in inst.truth.items():
            if eid not in got:
                defects.append(Defect("missing_email", "critical", eid))
                continue
            g = got[eid]
            acts = g.get("actions") if isinstance(g.get("actions"), list) else []
            got_types = sorted(str(x.get("type", "")).strip().lower() for x in acts
                               if isinstance(x, dict))
            want_types = sorted(x["type"] for x in t["actions"])
            if got_types != want_types:
                defects.append(Defect("wrong_action_set", "critical",
                                      f"{eid}: {got_types} vs {want_types}"))
            else:
                want_amt = {x["type"]: x["amount"] for x in t["actions"] if x["amount"] > 0}
                for x in acts:
                    ty = str(x.get("type", "")).strip().lower()
                    if ty in want_amt:
                        v = as_float(x.get("amount"))
                        if v is None or abs(v - want_amt[ty]) > 0.5:
                            defects.append(Defect("wrong_amount", "major",
                                                  f"{eid}/{ty}: {v} vs {want_amt[ty]:.2f}"))
            reply = str(g.get("reply", "")).lower()
            missing = [c for c in t["cites"] if c.lower() not in reply]
            if missing:
                defects.append(Defect("missing_citation", "minor",
                                      f"{eid}: {','.join(missing)}"))
            hits = [p for p in t["forbidden"] if p in reply]
            if hits:
                defects.append(Defect("forbidden_promise", "critical",
                                      f"{eid}: '{hits[0]}'"))
        return defects
