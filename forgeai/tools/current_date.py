"""Today's date - the one fact no document in a static corpus can contain."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as _tz


def run(timezone: str = "", timezone_name: str = "UTC", **_ignored) -> str:
    name = (timezone or timezone_name or "UTC").strip()
    now_utc = datetime.now(tz=_tz.utc)
    lines = [
        f"UTC now: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} ({now_utc.strftime('%A')})",
        f"UTC date: {now_utc.date().isoformat()}",
    ]
    if name.upper() != "UTC":
        try:
            from zoneinfo import ZoneInfo

            local = now_utc.astimezone(ZoneInfo(name))
            lines.append(
                f"{name}: {local.strftime('%Y-%m-%d %H:%M:%S')} ({local.strftime('%A')})"
            )
        except Exception:
            lines.append(f"(unknown timezone '{name}', showing UTC only)")
    return "\n".join(lines)
