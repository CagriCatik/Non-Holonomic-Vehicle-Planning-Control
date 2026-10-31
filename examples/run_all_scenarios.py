"""Run all registered vehicle planning scenarios and export plots and GIF animations."""

from __future__ import annotations

from non_holonomic_vehicle_planning.cli import run_benchmark_suite

if __name__ == "__main__":
    run_benchmark_suite(replan=False, animate=True, output_dir="./plots")
