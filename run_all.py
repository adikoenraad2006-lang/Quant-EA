#!/usr/bin/env python3
"""Run all four layers in order.

    python run_all.py              # real data
    python run_all.py --synthetic  # random walks, to smoke-test the pipeline
"""

from __future__ import annotations

import subprocess
import sys

LAYERS = [
    "layer1_data_strategies.py",
    "layer2_sweep.py",
    "layer3_robustness.py",
    "layer4_cross_sectional.py",
]

if __name__ == "__main__":
    extra = sys.argv[1:]
    for script in LAYERS:
        print(f"\n\n>>> {script} {' '.join(extra)}\n")
        rc = subprocess.call([sys.executable, script, *extra])
        if rc != 0:
            sys.exit(rc)
