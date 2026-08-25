#!/usr/bin/env python3
"""Validate the configured manifest and analysis settings."""

from eyeassist.cli import main

raise SystemExit(main(["validate-data", "--config", "configs/analysis.yaml"]))
