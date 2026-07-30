# `logger` is deliberately not re-exported here: it needs `rich`, which is a
# training-time dependency. Importing it eagerly would make a core
# `pip install metro-asr` fail at import time. Import it directly instead:
#     from metro_asr.utils.logger import get_logger
import sys

from metro_asr.utils.config import load_config

__all__ = ["load_config", "enable_utf8_stdout"]


def enable_utf8_stdout():
    """
    Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page (cp1252) that cannot encode
    Arabic — or the emoji these scripts print — so any status line raises
    UnicodeEncodeError. Safe and a no-op everywhere else.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
