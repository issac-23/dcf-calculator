"""How a money amount is labelled. Nothing here converts between currencies."""

from __future__ import annotations

from typing import Optional


def currency_prefix(currency: Optional[str]) -> str:
    """`$` for USD or an unknown currency, otherwise the code and a space.

    Unknown stays `$` so US tickers and the `.info` fallback (which has no
    currency code) read exactly as they always have.
    """
    if not currency or currency.upper() == "USD":
        return "$"
    return f"{currency.upper()} "
