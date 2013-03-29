#!/usr/bin/env python2
"""
    Runs the executable of the watermarking module.
"""

if __name__ == "__main__":
    from watermark.bin import run
    run(prefix=[-1, 1])