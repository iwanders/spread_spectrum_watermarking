#!/usr/bin/env python3
"""
    Runs the executable of the watermarking module.
"""


import hashlib
import base64

word = "IWanders"
salt = [33, 162, 183, 206, 243, 68, 67, 89]#[random.randint(0,255) for i in range(0,8)]

repeat = 5 # Number of times to repeat this (actually times two)

infront = bytearray(bytes(word, 'ascii')) + bytes(salt)

m = hashlib.md5()
m.update(infront)

infront = m.digest()



wmprefix = []

# convert this into numbers.
sign = 1
for i in range(repeat*2):
    for letter in infront:
        wmprefix.append(sign*((letter)/128.0))
        sign = -sign # flip it every letter
    if (len(infront)%2 == 0):
        sign = -sign # also flip it if the word is of even length

average = sum(wmprefix)
# print("average: %f" % average)
# print(wmprefix)
# print("Length: %f" % len(wmprefix))

import sys
sys.path.append("/home/ivor/Documents/Code/python/watermarking/src/")

if __name__ == "__main__":
    from watermark.bin import run
    run(prefix=wmprefix, wm_add_entries={"prefix_word":word, "prefix_salt":salt, "prefix_md5_hash_b64": str(base64.b64encode(infront)), "prefix_wmprefix":wmprefix})
