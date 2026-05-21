"""Pre-flight FRED series validation.

Iterates every series ID declared in config/features.yaml, attempts a FRED
pull, and fails the run with a clear error summary if any series is empty or
unavailable.  Run this before training any model.

Usage
-----
    python -m src.data.validate_fred
"""
import logging
import sys
from pathlib import Path

import yaml

from src.data.fred_loader import load_series
from src.utils.seeds import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/features.yaml")


def _collect_all_series_ids(config: dict) -> set[str]:
    """Extract every unique FRED series ID from the features config."""
    ids: set[str] = set()

    # Universal block
    for sid in config.get("universal_macro", {}).get("fred", []):
        ids.add(sid)

    # Sector-specific blocks
    for _etf, block in config.get("sector_features", {}).items():
        for sid in block.get("fred", []):
            ids.add(sid)

    return ids


def validate_all(force_refresh: bool = False) -> dict[str, bool]:
    """Validate every FRED series in features.yaml.

    Parameters
    ----------
    force_refresh:
        Re-pull from FRED even if cached.

    Returns
    -------
    dict[str, bool]
        Mapping series_id -> True (ok) / False (failed).

    Raises
    ------
    SystemExit
        If any series fails validation.
    """
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    all_ids = _collect_all_series_ids(config)
    logger.info("Validating %d unique FRED series IDs...", len(all_ids))

    results: dict[str, bool] = {}
    failed: list[str] = []

    for sid in sorted(all_ids):
        try:
            s = load_series(sid, force_refresh=force_refresh)
            n = len(s.dropna())
            logger.info("  OK  %s  (%d non-null observations)", sid, n)
            results[sid] = True
        except Exception as exc:
            logger.error("  FAIL %s: %s", sid, exc)
            results[sid] = False
            failed.append(f"  - {sid}: {exc}")

    # Print summary
    n_ok = sum(results.values())
    n_fail = len(failed)
    logger.info("\n=== FRED Validation Summary ===")
    logger.info("  Passed : %d / %d", n_ok, len(results))

    if failed:
        logger.error("  Failed : %d / %d", n_fail, len(results))
        for msg in failed:
            logger.error(msg)
        logger.error(
            "\nFATAL: %d FRED series failed validation. "
            "Fix or remove these series IDs from config/features.yaml before continuing.",
            n_fail,
        )
        sys.exit(1)

    logger.info("All FRED series validated successfully.")
    return results


if __name__ == "__main__":
    set_all_seeds()
    validate_all()
