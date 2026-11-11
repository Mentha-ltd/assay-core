# Gate for buyers: the package opens no connections, runs no shell commands,
# and imports only the four packages we declare.
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "assay_core"
ALLOWED = {"assay_core", "numpy", "pandas", "pydantic", "duckdb"}
BANNED_MODULES = {
    "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "ssl",
    "ftplib", "smtplib", "telnetlib", "xmlrpc", "subprocess", "multiprocessing",
}  # fmt: skip
BANNED_CALLS = {"eval", "exec", "system", "popen", "__import__"}


def _modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            yield f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")


def _trees():
    files = sorted(SRC.rglob("*.py"))
    assert files, "no source files found"
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in files]


def test_no_network_or_shell_modules():
    bad = [(p.name, m) for p, t in _trees() for m in _modules(t) if m in BANNED_MODULES]
    assert not bad, f"network or shell modules imported: {bad}"


def test_only_declared_third_party_packages():
    found = {m for _p, t in _trees() for m in _modules(t)}
    outside = {m for m in found if m not in sys.stdlib_module_names} - ALLOWED
    assert not outside, f"undeclared third-party imports: {sorted(outside)}"


def test_no_dynamic_code_or_shell_calls():
    bad = [(p.name, c) for p, t in _trees() for c in _calls(t) if c in BANNED_CALLS]
    assert not bad, f"dynamic code or shell calls: {bad}"
