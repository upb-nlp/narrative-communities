#!/usr/bin/env python3
"""Run GraphRAG indexing."""

import runpy

if __name__ == "__main__":
    runpy.run_module("narrative_map.indexing.server", run_name="__main__")
