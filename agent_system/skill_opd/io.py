from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable


def to_jsonable(value: Any) -> Any:
    """Convert common rollout objects to plain JSON-compatible values."""

    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return str(value)


class JsonlWriter:
    """Append records to a JSONL file."""

    def __init__(self, path: str | Path, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.path.write_text("", encoding="utf-8")

    def write_many(self, records: Iterable[Any]) -> int:
        count = 0
        with self.path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(to_jsonable(record), ensure_ascii=False))
                file.write("\n")
                count += 1
        return count

