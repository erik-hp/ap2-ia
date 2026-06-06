"""Configuração compartilhada da suíte de testes."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")
