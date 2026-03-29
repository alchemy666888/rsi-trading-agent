from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator


@contextmanager
def trace_span(name: str, trace_dir: str = "traces") -> Generator[dict[str, Any], None, None]:
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    trace: dict[str, Any] = {"name": name, "steps": []}
    try:
        yield trace
    finally:
        path = Path(trace_dir) / f"{name}_{os.urandom(4).hex()}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def write_month_trace(month_id: str, payload: dict[str, Any], trace_dir: str = "traces") -> str:
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    safe_month = month_id.replace("/", "-")
    path = Path(trace_dir) / f"month_{safe_month}.json"
    body = {
        "month_id": month_id,
        "generated_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_run_manifest(payload: dict[str, Any], trace_dir: str = "traces") -> str:
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = Path(trace_dir) / f"run_manifest_{stamp}.json"
    body = {"generated_at": datetime.now(UTC).isoformat(), **payload}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
