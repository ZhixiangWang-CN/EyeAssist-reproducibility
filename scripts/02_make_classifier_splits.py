#!/usr/bin/env python3
"""Generate the 50 shared case partitions."""

from eyeassist.cli import main

raise SystemExit(
    main(
        [
            "make-splits",
            "--config",
            "configs/analysis.yaml",
            "--analysis",
            "classifier",
            "--test-groups",
            "15",
            "--output",
            "outputs/classifier_splits.csv",
        ]
    )
)
