"""Shared pytest configuration.

Adds a ``network`` marker for tests that hit live USGS web services. They are
skipped by default so the suite runs offline and deterministically; enable them
with ``pytest --run-network``.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' that call the live USGS services",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test calls the live USGS web services (opt-in)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network to hit live USGS services")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
