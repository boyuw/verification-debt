"""Family 3 - data cleansing / record consolidation.

Twelve to sixteen messy records that resolve to four entities.  Records carry no
entity key: the producer has to cluster them by normalised identity, normalise
formats, and defer to a system-of-record table where it speaks.  One planted trap
is a near-duplicate pair that must *not* be merged (same surname, different
person); one planted contradiction is a field where the messy record disagrees
with the system of record.

Verifiability is high on the fields and lower on the clustering: a merged pair is
cheap to confirm once found, but deciding whether two records are the same person
requires judgement the rubric can only partly specify.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from .base import Defect, Instance, TaskFamily

FIRST = ["Alicia", "Bram", "Chen", "Dalia", "Emeka", "Freya", "Goran", "Hana", "Iker", "Jonas"]
LAST = ["Weber", "Alvarez", "Nakamura", "Osei", "Kovac", "Lindgren", "Haddad", "Moreau"]
CITIES = ["Springfield", "Shelbyville", "Ogdenville", "North Haverbrook", "Capital City"]
PLANS = ["basic", "pro", "enterprise"]


class CleanseFamily(TaskFamily):
    name = "cleansing"
    label = "Data cleansing / record consolidation"
    vis = {"ground_truth_checkability": 3, "rule_determinism": 2, "judgement": 2}
    rubric = (
        "Check the consolidated record set against the raw records and the system of record.\n"
        "  (1) one output record per real entity: duplicates merged, near-duplicates that are "
        "different people kept apart.\n"
        "  (2) source_ids: exactly the raw ids that belong to that entity, no more, no fewer.\n"
        "  (3) name, email, signup_date normalised as the rules require (title case, lowercase "
        "email, ISO date).\n"
        "  (4) city and plan taken from the system of record wherever it lists the entity, even "
        "when a raw record disagrees.\n"
        "A product is accepted only if every entity is present, correctly clustered, and every "
        "field matches."
    )
    checklist = [
        "Every raw record id appears in exactly one output record's source_ids.",
        "No two output records share the same normalised email.",
        "Near-duplicate names with different emails were checked against the system of record before merging.",
        "Every email is lowercase with no spaces; every date is ISO YYYY-MM-DD; every name is title case.",
        "City and plan were taken from the system of record wherever it lists the entity.",
        "No field was left blank when a non-blank value existed in any merged record.",
    ]

    # ------------------------------------------------------------------ generate
    def generate(self, n: int, seed: int = 0) -> List[Instance]:
        rng = random.Random(seed * 3301 + 7)
        return [self._one(rng, i) for i in range(n)]

    @staticmethod
    def _fmt_date(rng: random.Random, y: int, m: int, d: int) -> str:
        style = rng.randrange(4)
        return [f"{y}-{m:02d}-{d:02d}", f"{m:02d}/{d:02d}/{str(y)[2:]}",
                f"{d} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} {y}",
                f"{m:02d}.{d:02d}.{y}"][style]

    def _one(self, rng: random.Random, idx: int) -> Instance:
        n_ent = 4
        people = []
        used_emails = set()
        for i in range(n_ent):
            fn, ln = rng.choice(FIRST), rng.choice(LAST)
            email = f"{fn.lower()}.{ln.lower()}{rng.randint(1, 99)}@example.com"
            while email in used_emails:
                email = f"{fn.lower()}.{ln.lower()}{rng.randint(100, 999)}@example.com"
            used_emails.add(email)
            people.append({
                "name": f"{fn} {ln}", "email": email,
                "y": rng.randint(2023, 2025), "m": rng.randint(1, 12), "d": rng.randint(1, 28),
                "city": rng.choice(CITIES), "plan": rng.choice(PLANS),
            })
        # trap: a fifth person who shares a surname with person 0 but is distinct
        trap = {
            "name": f"{rng.choice([f for f in FIRST if f != people[0]['name'].split()[0]])} "
                    f"{people[0]['name'].split()[1]}",
            "email": f"x.{people[0]['name'].split()[1].lower()}{rng.randint(10, 99)}@example.com",
            "y": rng.randint(2023, 2025), "m": rng.randint(1, 12), "d": rng.randint(1, 28),
            "city": rng.choice(CITIES), "plan": rng.choice(PLANS),
        }

        raw: List[Dict[str, Any]] = []
        rid = 1
        for pi, p in enumerate(people):
            k = [3, 2, 1, 2][pi % 4]
            for j in range(k):
                nm = p["name"]
                style = rng.randrange(3)
                nm = nm.upper() if style == 0 else (nm.lower() if style == 1 else nm)
                em = p["email"]
                if j == 1 and rng.random() < 0.5:
                    em = em.upper()
                if j == 2 and rng.random() < 0.4:
                    em = " " + em
                if j == k - 1 and k > 1 and rng.random() < 0.3:
                    em = ""                                    # blank in one record
                city = p["city"]
                plan = p["plan"]
                if j == 1:
                    city = city.upper() if rng.random() < 0.5 else city.lower()
                if j == 2:
                    plan = plan.upper()
                raw.append({"rid": f"R{rid:02d}", "name": nm, "email": em,
                            "date": self._fmt_date(rng, p["y"], p["m"], p["d"]),
                            "city": city, "plan": plan,
                            "phone": f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}",
                            "entity": pi})
                rid += 1
        # distinct person who shares a surname
        raw.append({"rid": f"R{rid:02d}", "name": trap["name"].upper(), "email": trap["email"],
                    "date": self._fmt_date(rng, trap["y"], trap["m"], trap["d"]),
                    "city": trap["city"], "plan": trap["plan"],
                    "phone": f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}",
                    "entity": n_ent})
        rid += 1

        # planted contradiction: one raw record disagrees with the system of record on city
        victim = rng.choice([r for r in raw if r["entity"] < n_ent - 1])
        alt = [c for c in CITIES if c != people[victim["entity"]]["city"]]
        victim["city"] = rng.choice(alt)
        victim["contradiction"] = True

        rng.shuffle(raw)
        sor = []
        for pi, p in enumerate(people):
            if pi == 3:                                        # one entity absent from SoR
                continue
            sor.append({"email": p["email"], "city": p["city"], "plan": p["plan"]})

        truth: Dict[str, Any] = {}
        for pi, p in enumerate(people):
            ids = sorted([r["rid"] for r in raw if r["entity"] == pi])
            truth[p["email"]] = {
                "name": p["name"], "email": p["email"],
                "signup_date": f"{p['y']}-{p['m']:02d}-{p['d']:02d}",
                "city": p["city"], "plan": p["plan"], "source_ids": ids,
            }
        truth[trap["email"]] = {
            "name": trap["name"], "email": trap["email"],
            "signup_date": f"{trap['y']}-{trap['m']:02d}-{trap['d']:02d}",
            "city": trap["city"], "plan": trap["plan"],
            "source_ids": [r["rid"] for r in raw if r["entity"] == n_ent],
        }

        text = self._render(raw, sor)
        canonical = json.dumps(
            {"records": [{**v, "source_ids": v["source_ids"]} for v in truth.values()]}, indent=1)
        return Instance(family=self.name, iid=f"{self.name}-{idx:04d}", text=text,
                        truth=truth, n_checks=6 * len(truth), canonical=canonical,
                        meta={"n_raw": len(raw), "n_entities": len(truth)})

    def _render(self, raw: List[Dict[str, Any]], sor: List[Dict[str, Any]]) -> str:
        lines = ["RAW CONTACT RECORDS (unconsolidated, arbitrary order)", ""]
        for r in raw:
            lines.append(f"{r['rid']} | name: {r['name']} | email: {r['email'] or '(blank)'} | "
                         f"signup: {r['date']} | city: {r['city']} | plan: {r['plan']} | "
                         f"phone: {r['phone']}")
        lines += ["", "SYSTEM OF RECORD (authoritative for the fields it lists)", ""]
        for s in sor:
            lines.append(f"email: {s['email']} | city: {s['city']} | plan: {s['plan']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ prompts
    def producer_system(self) -> str:
        return (
            "You are a data steward consolidating a contact list. Apply these rules exactly.\n"
            "  N1  Normalise: names to Title Case, emails to lowercase with whitespace removed, "
            "dates to ISO YYYY-MM-DD, plan to lowercase.\n"
            "  N2  Two records describe the same entity when their normalised emails match. If a "
            "record has no email, match it on normalised name plus city. Records that merely "
            "share a surname are different people unless an email matches.\n"
            "  N3  Where the system of record lists an entity (matched by normalised email), its "
            "city and plan override any conflicting raw record.\n"
            "  N4  Output exactly one record per entity, keyed by the normalised email. "
            "source_ids lists every raw id merged into that entity, sorted ascending.\n"
            "  N5  Never leave a field blank if any merged record supplies a value.\n\n"
            "Respond with JSON only:\n"
            '{"records":[{"name":"...","email":"...","signup_date":"YYYY-MM-DD","city":"...",'
            '"plan":"basic|pro|enterprise","source_ids":["R01","R02"]}]}'
        )

    def verifier_system(self) -> str:
        return (
            "You are an independent data-quality reviewer. You did not consolidate these records. "
            "Re-derive the correct consolidated set yourself from the raw records and the system "
            "of record, then compare it to the product.\n\n"
            "Respond with JSON only:\n"
            '{"accepted":true|false,"checks":<int>,"defects":[{"email":"...","field":"cluster|source_ids|name|email|'
            'signup_date|city|plan","severity":"critical|major|minor",'
            '"expected":"<your value>"}]}\n'
            "Set accepted to false if any entity is missing, any cluster is wrong, or any field "
            "value is wrong."
        )

    # ------------------------------------------------------------------ grading
    def census(self, inst: Instance, product: Optional[dict]) -> List[Defect]:
        if not product:
            return [Defect("unparseable_product", "critical", "no JSON product returned")]
        rows = product.get("records")
        if not isinstance(rows, list):
            return [Defect("missing_records", "critical", "no records array")]
        got: Dict[str, dict] = {}
        for r in rows:
            if isinstance(r, dict) and r.get("email"):
                got[str(r["email"]).strip().lower().replace(" ", "")] = r

        defects: List[Defect] = []
        for email, t in inst.truth.items():
            key = email.lower()
            if key not in got:
                # maybe merged into another record's source_ids
                merged = any(t["source_ids"][0] in (g.get("source_ids") or []) for g in got.values())
                sev = "major" if merged else "critical"
                defects.append(Defect("missing_entity" if not merged else "over_merged",
                                      sev, email))
                continue
            g = got[key]
            sids = g.get("source_ids") if isinstance(g.get("source_ids"), list) else []
            sids = sorted(str(s).strip().upper() for s in sids)
            if sids != sorted(t["source_ids"]):
                sev = "critical" if len(sids) < len(t["source_ids"]) else "major"
                defects.append(Defect("wrong_cluster", sev,
                                      f"{email}: {sids} vs {t['source_ids']}"))
            for fld, sev in (("name", "major"), ("signup_date", "major"),
                             ("city", "critical"), ("plan", "critical")):
                v = str(g.get(fld, "")).strip()
                if fld == "plan":
                    v = v.lower()
                if v != t[fld]:
                    defects.append(Defect(f"bad_{fld}", sev, f"{email}: {v!r} vs {t[fld]!r}"))
        # an extra record that is really a split duplicate
        for key in got:
            if key not in {e.lower() for e in inst.truth}:
                defects.append(Defect("spurious_entity", "major", key))
        return defects
