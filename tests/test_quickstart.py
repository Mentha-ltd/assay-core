import runpy
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_script_runs(capsys):
    runpy.run_path(str(QUICKSTART))
    out = capsys.readouterr().out
    assert "record intact: True" in out
    assert "90% range" in out
    assert "group weather = windy" in out  # explain found the planted cause
