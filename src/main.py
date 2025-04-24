#!/usr/bin/env python3
"""
Entry-point with opt-in verbose logging (off by default).
"""

from __future__ import annotations

import argparse
import logging

from . import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rent-roll summarizer")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable detailed DEBUG logging (default: no logging)",
    )
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(
            level=logging.DEBUG,                        # ← verbose
            format="%(asctime)s %(levelname)s %(name)s : %(message)s",
        )
        logging.debug("Verbose logging enabled")
    else:
        # Disable everything up to CRITICAL => effectively silent
        logging.disable(logging.CRITICAL)

    pipeline.main()


if __name__ == "__main__":
    main()
