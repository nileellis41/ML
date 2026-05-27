"""Global seed management -- call set_all_seeds() at every entrypoint.

Also loads .env from the project root (if present) so API keys are always
available without manually exporting them in the shell.
"""
import os
import random
import logging
from pathlib import Path

import numpy as np

# Auto-load .env — works with or without python-dotenv, from any working directory.
def _load_env() -> None:
    """Parse .env at the project root; never overwrite vars already in the shell."""
    # seeds.py lives at <root>/src/utils/seeds.py → root is two parents up
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Fallback: plain parse — no third-party dependency needed
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_env()

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for numpy, torch, and python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: raise a warning (not an error) for ops without
        # a deterministic kernel — safe for CPU-only LSTM workloads.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        logger.warning("torch not installed; skipping torch seed")

    logger.info("All random seeds set to %d", seed)
