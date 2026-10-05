"""Lets the lessons run from anywhere and tells you whether Jev is live or mocked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retail_mesh import config  # noqa: E402
from retail_mesh.jev_client import get_jev  # noqa: E402,F401

print(f"[Jev is {'MOCKED (set TYPESAFE_API_KEY for the real model)' if config.use_mock_jev() else 'LIVE'}]\n")
