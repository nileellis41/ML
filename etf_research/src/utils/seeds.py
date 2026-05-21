"""Global seed management -- call set_all_seeds() at every entrypoint.

Also loads .env from the project root (if present) so API keys are always
available without manually exporting them in the shell.
"""
import os
import random
import logging
from pathlib import Path

import numpy as np

# Auto-load .env on import so all modules that call set_all_seeds() get keys
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)  # override=False: shell vars take precedence
except ImportError:
    pass  # python-dotenv optional; keys can still be set in the shell

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
    except ImportError:
        logger.warning("torch not installed; skipping torch seed")

    logger.info("All random seeds set to %d", seed)
