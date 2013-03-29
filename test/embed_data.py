#!/usr/bin/env python2

# import the module
try:
    from watermark import cox
except ImportError:
    print("Module not installed. Using one in ../")
    import sys
    sys.path.insert(0, "..")
    from watermark import cox


# seed the number generator, always the same results.
import random
random.seed(0)

ourLength = 2000
key = [random.gauss(0,1) for x in range(0,ourLength)]
data  = [random.choice((-1,1)) for x in range(0,ourLength)]
#
embedthis = [key[i] * data[i] for i in range(0,ourLength)]
print(embedthis[0:40])

input_path="Lenna.bmp"
output_file="watermarked.png"
input_file = cox.yiq_dct_image.open(input_path)
mark = cox.dctwatermarker(input_file)
mark.wm(embedthis)
mark.embed()
mark.output().write(output_file)


target_image = cox.yiq_dct_image.open(output_file)
tester = cox.dctwatermarker(target_image)
tester.orig_file(input_path)

extract = tester.extract(size=ourLength)

print(extract[0:40])
sanitized = [-1 if (extract[i] * key[i])<0 else 1 for i in range(0,ourLength)]

print(sanitized[0:40])
print(data[0:40])

