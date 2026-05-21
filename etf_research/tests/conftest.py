"""Pytest configuration for ETF research tests.

Adds the project root to sys.path so imports work correctly from all test files.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
