import sys

from .cli import main

# Консоль Windows может не уметь показать отдельные символы (Ø, ≤): лучше «?», чем падение
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

raise SystemExit(main())
