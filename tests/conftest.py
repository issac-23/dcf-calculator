"""Keep every test offline by pointing the app at the recorded market source."""

from __future__ import annotations

import pytest
import streamlit as st

from dcf.market import SOURCE_ENV_VAR


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Nothing in this suite touches the network.

    The app picks its market-data source at startup from DCF_MARKET_SOURCE, so
    the whole suite sets it once here. The recorded source serves the fixtures
    and a pinned 10-year Treasury; a journey that needs a different quote sets
    E2E_TREASURY_CLOSE with monkeypatch.setenv.
    """
    monkeypatch.setenv(SOURCE_ENV_VAR, "tests.e2e.replay:recorded_source")
    st.cache_data.clear()
    yield
    st.cache_data.clear()
