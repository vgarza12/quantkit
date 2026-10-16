"""Tamper-evident audit log: each JSONL record includes the hash of the previous one.

Editing or deleting any past line breaks the chain, which `AuditLog.verify` detects.
Sensitive-looking fields are redacted before anything is written.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

_SENSITIVE = re.compile(r"(?i)key|secret|token|passw")
GENESIS = "0" * 64


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if _SENSITIVE.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self._last = GENESIS
        if self.path.exists():
            lines = self.path.read_text().strip().splitlines()
            if lines:
                self._last = json.loads(lines[-1])["hash"]

    def append(self, event: dict) -> dict:
        record = {"ts": time.time(), "event": redact(event), "prev": self._last}
        record["hash"] = _digest(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._last = record["hash"]
        return record

    @staticmethod
    def verify(path: str) -> tuple[bool, int | None]:
        """(True, None) if intact, else (False, line_number_of_first_bad_record)."""
        prev = GENESIS
        for i, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, i
            if rec.get("prev") != prev or rec.get("hash") != _digest(rec):
                return False, i
            prev = rec["hash"]
        return True, None
