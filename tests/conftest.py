import os
import sys
from pathlib import Path

# tests never hit the network
os.environ["JEV_MOCK"] = "1"
os.environ["LLM_MOCK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
