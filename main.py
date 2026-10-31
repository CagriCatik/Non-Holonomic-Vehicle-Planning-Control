"""Main entry point for Non-Holonomic Vehicle Planning & Control."""

from __future__ import annotations

import sys

from non_holonomic_vehicle_planning.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
