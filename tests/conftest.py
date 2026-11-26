"""Keep the test suite hermetic.

A developer with a real `.env` would otherwise have the end-to-end tests call a
live model: slow, costly, rate-limited and non-deterministic. The suite's job
is to test the factory's machinery, not a provider, so every test runs in the
deterministic no-model mode regardless of local configuration.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _force_deterministic_mode(monkeypatch):
    monkeypatch.setenv("FORGEAI_PROVIDER", "heuristic")
    monkeypatch.setenv("FORGEAI_API_KEY", "")
    monkeypatch.setenv("FORGEAI_BASE_URL", "")
    monkeypatch.setenv("FORGEAI_MODEL", "heuristic-v1")
    yield
