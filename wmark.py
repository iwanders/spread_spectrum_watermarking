#!/usr/bin/env python2
"""
    Runs the executable of the watermarking module.
"""

word = "Ivor Wanders"
repeat = 10 # If this is even the prefix is exactly averaged to 0.

wmprefix = []

# convert this into numbers.
sign = 1
for i in range(repeat):
    for letter in word:
        wmprefix.append(sign*(ord(letter)/128.0))
        sign = -sign # flip it every letter
    if (len(word)%2 == 0):
        sign = -sign # also flip it if the word is of even length

#average = sum(wmprefix)
#print("average: %f" % average)
#print(wmprefix)


if __name__ == "__main__":
    from watermark.bin import run
    run(prefix=wmprefix)