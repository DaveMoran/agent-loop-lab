"""Injectable faults. Turn them on with SHOP_FAULTS=slow_search,checkout_timeout,double_decrement"""
import os

KNOWN = {"slow_search", "checkout_timeout", "double_decrement"}


def active() -> set[str]:
    raw = os.environ.get("SHOP_FAULTS", "")
    found = {f.strip() for f in raw.split(",") if f.strip()}
    unknown = found - KNOWN
    if unknown:
        raise ValueError(f"unknown faults: {sorted(unknown)}; known: {sorted(KNOWN)}")
    return found
