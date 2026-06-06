"""Testes dos critérios de checkpoint do treino."""

from train import _is_improved


def test_checkpoint_improvement_modes() -> None:
    assert _is_improved(1.0, None, "min")
    assert _is_improved(0.9, 1.0, "min")
    assert not _is_improved(1.1, 1.0, "min")
    assert _is_improved(0.8, 0.7, "max")
    assert not _is_improved(0.6, 0.7, "max")
