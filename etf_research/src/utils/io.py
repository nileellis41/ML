"""I/O helpers: parquet read/write with consistent dtypes and logging."""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save DataFrame to parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    logger.debug("Saved %s rows to %s", len(df), path)


def load_parquet(path: str | Path, **kwargs) -> Optional[pd.DataFrame]:
    """Load parquet file; return None (with warning) if file does not exist."""
    path = Path(path)
    if not path.exists():
        logger.warning("Cache miss: %s does not exist", path)
        return None
    df = pd.read_parquet(path, **kwargs)
    logger.debug("Loaded %s rows from %s", len(df), path)
    return df


def cache_hit(path: str | Path) -> bool:
    """Return True if a valid parquet cache file exists at path."""
    return Path(path).exists() and Path(path).stat().st_size > 0
