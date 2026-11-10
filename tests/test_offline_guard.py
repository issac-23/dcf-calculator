"""The guard has to catch what yfinance really does, not just what urllib does.

Both calls below run with the network up on a developer machine, so "caught"
means the guard stopped them, not that the connection happened to fail.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest
import yfinance

pytest_plugins = ["pytester"]


def test_urllib_is_recorded(network_guard):
    with pytest.raises(OSError):
        urllib.request.urlopen("https://example.com", timeout=5)

    assert network_guard.blocked, "urllib got past the guard"
    assert "example.com" in network_guard.blocked[0]
    network_guard.blocked.clear()  # expected here; keep teardown green


def test_yfinance_is_recorded(network_guard):
    # Whether the exception surfaces or is swallowed, the record is the evidence.
    try:
        yfinance.Ticker("AAPL").info
    except Exception:
        pass

    assert network_guard.blocked, "yfinance got past the guard"
    assert any("yahoo" in attempt for attempt in network_guard.blocked)
    network_guard.blocked.clear()


def test_swallowed_attempts_fail_and_live_is_exempt(pytester, monkeypatch):
    """One subprocess run covers both, since starting pytest with Streamlit is slow."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent.parent))
    pytester.makeconftest("from tests.offline_guard import network_guard  # noqa: F401")
    pytester.makepyfile(
        """
        import socket
        import urllib.request

        import pytest

        def test_swallows_it():
            try:
                urllib.request.urlopen("https://example.com", timeout=5)
            except Exception:
                pass

        @pytest.mark.live
        def test_live_has_no_guard():
            assert type(socket.socket.connect).__name__ == "method_descriptor"
        """
    )
    pytester.makeini("[pytest]" + chr(10) + "markers = live")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["*test tried to reach the network*example.com*"])
