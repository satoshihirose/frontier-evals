from __future__ import annotations

from typing import NamedTuple


class ReproductionAttempt(NamedTuple):
    use_py3_11: bool
    make_venv: bool


# Keep this order shared by formal reproduction and execution-feedback diagnostics.
REPRODUCTION_SALVAGE_ATTEMPTS = (
    ReproductionAttempt(use_py3_11=False, make_venv=False),
    ReproductionAttempt(use_py3_11=True, make_venv=False),
    ReproductionAttempt(use_py3_11=False, make_venv=True),
    ReproductionAttempt(use_py3_11=True, make_venv=True),
)
