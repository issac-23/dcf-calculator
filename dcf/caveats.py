"""The ways a valuation is weaker than it looks, as typed findings.

These used to be wording scattered across the stack: warning strings built in
dcf/data.py, a sector flag, the clamped set, `debt_cost_unknown` on the WACC
breakdown, a None implied growth, each of which app.py and the E2E report
phrased separately -- which is how the report came to print J.P. Morgan's
wrong-sector caveat twice. Here they are found once, from facts the valuation
run already holds, and carried on the `Valuation`. Each surface words each
caveat once, where it belongs on its own page.

Nothing here touches Streamlit or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from dcf.data import CompanyData
from dcf.wacc import WACCBreakdown


class CaveatKind(Enum):
    # Declaration order is the order a report lists them in.
    WRONG_SECTOR = "wrong_sector"
    DEFAULT_CLAMPED = "default_clamped"
    NO_IMPLIED_GROWTH = "no_implied_growth"
    DEBT_UNREPORTED = "debt_unreported"
    NEGATIVE_MARGIN = "negative_margin"
    WACC_NOT_DERIVED = "wacc_not_derived"
    DEBT_COST_UNKNOWN = "debt_cost_unknown"


@dataclass(frozen=True)
class Caveat:
    kind: CaveatKind
    # What the finding is about, when that varies: the sector for
    # WRONG_SECTOR, the assumption key for DEFAULT_CLAMPED, the reason for
    # WACC_NOT_DERIVED. Empty otherwise.
    subject: str = ""


def find_caveats(
    company: CompanyData,
    wacc: Optional[WACCBreakdown],
    wacc_reason: Optional[str],
    clamped: Iterable[str],
    implied_found: Optional[bool],
) -> tuple[Caveat, ...]:
    """`implied_found` is None when there was no price to solve against."""
    found = []
    if company.is_dcf_inappropriate:
        found.append(Caveat(CaveatKind.WRONG_SECTOR, company.sector))
    for key in sorted(clamped):
        found.append(Caveat(CaveatKind.DEFAULT_CLAMPED, key))
    if implied_found is False:
        found.append(Caveat(CaveatKind.NO_IMPLIED_GROWTH))
    if company.total_debt is None:
        found.append(Caveat(CaveatKind.DEBT_UNREPORTED))
    margin = company.historical_operating_margin
    if margin is not None and margin < 0:
        found.append(Caveat(CaveatKind.NEGATIVE_MARGIN))
    if wacc is None:
        found.append(Caveat(CaveatKind.WACC_NOT_DERIVED, wacc_reason or ""))
    elif wacc.debt_cost_unknown:
        found.append(Caveat(CaveatKind.DEBT_COST_UNKNOWN))
    return tuple(found)
