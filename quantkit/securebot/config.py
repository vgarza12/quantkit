"""Secrets handling: environment-only, never printed, plus a scanner for hardcoded keys."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


class MissingSecret(RuntimeError):
    pass


@dataclass(frozen=True)
class Secrets:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    def __repr__(self) -> str:  # never leak values via logs/tracebacks/debuggers
        return "Secrets(api_key=***, api_secret=***)"

    __str__ = __repr__


def load_secrets(env=None) -> Secrets:
    """Read QK_API_KEY / QK_API_SECRET from the environment. Fail loudly if absent."""
    env = os.environ if env is None else env
    missing = [k for k in ("QK_API_KEY", "QK_API_SECRET") if not env.get(k)]
    if missing:
        raise MissingSecret(f"Set environment variables: {', '.join(missing)}")
    return Secrets(api_key=env["QK_API_KEY"], api_secret=env["QK_API_SECRET"])


_PATTERN = re.compile(r"""(?ix)(api[_-]?key|secret|token|passw(or)?d)\s*[:=]\s*['"][A-Za-z0-9/+_\-]{16,}['"]""")


def scan_for_secrets(root: str, exts=(".py", ".env", ".json", ".yaml", ".yml", ".txt", ".md")) -> list[tuple[str, int]]:
    """Return (file, line_no) for lines that look like hardcoded credentials."""
    hits = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix in exts and ".git" not in p.parts:
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if _PATTERN.search(line):
                        hits.append((str(p), i))
            except OSError:
                continue
    return hits
