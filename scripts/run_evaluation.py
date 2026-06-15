#!/usr/bin/env python3
"""Run the full prediction and evaluation pipeline."""

import runpy

if __name__ == "__main__":
    runpy.run_module("narrative_map.prediction.predict", run_name="__main__")
