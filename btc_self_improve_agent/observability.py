from __future__ import annotations

import json
import os
from contextlib import contextmanager
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
