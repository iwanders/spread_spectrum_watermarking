#!/usr/bin/env python2
"""
    Runs the executable of the watermarking module.
"""

if __name__ == "__main__":
    import watermark.bin
    watermark.bin.prefixWatermark([-1, 1])
    watermark.bin.run()