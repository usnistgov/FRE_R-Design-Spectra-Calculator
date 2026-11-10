"""Root pytest configuration.

This file exists for its location rather than its contents. pytest puts the
directory of every ``conftest.py`` on ``sys.path``, and that is what lets the
tests import the web app's own modules -- ``map_view`` and ``app.py`` sit at
the repository root and, unlike ``frer``, are not installed by
``pip install -e .`` (see ``[tool.setuptools.packages.find]`` in
pyproject.toml, which packages only ``src``).

Without it the suite passes under ``python -m pytest``, which happens to put
the working directory on the path, and fails under plain ``pytest`` -- which
is what CI runs.

Shared fixtures and the ``network`` marker live in tests/conftest.py.
"""
