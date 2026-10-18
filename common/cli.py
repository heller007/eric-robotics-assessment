"""Small wrapper so a missing dataset or an uninstalled package prints a message
instead of a traceback. The message is the useful part; the stack is not."""
from __future__ import annotations

import functools


def friendly(main):
    @functools.wraps(main)
    def wrapper(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except KeyboardInterrupt:
            raise SystemExit("\ninterrupted")
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(f"\nerror: {exc}")

    return wrapper
