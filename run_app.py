"""Launch the Streamlit research UI in the default browser."""

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
