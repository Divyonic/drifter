"""Repo-root launcher shim.

The real Streamlit app lives in the installable package at ``cdm/app.py`` so it
ships with ``pip install``. This shim keeps ``streamlit run app.py`` working from a
checkout. The ``drifter`` command (see ``cdm/cli.py``) runs the packaged app directly.
"""

from cdm.app import main

main()
