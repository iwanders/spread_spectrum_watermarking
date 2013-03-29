#!/usr/bin/env python2



# import the module
try:
    from watermark import cox
except ImportError:
    print("Module not installed. Using one in ../")
    import sys
    sys.path.insert(0, "..")
    from watermark import cox

# check if we have a Lenna.bmp, otherwise create it.
import os
if (not os.path.lexists('Lenna.bmp')):
    print("Lenna does not exist, making it.")
    import scipy
    # save Lenna.bmp as color image (even though it is BW)
    scipy.misc.imsave("Lenna.bmp", scipy.dstack((scipy.misc.lena(),
                                                scipy.misc.lena(),
                                                    scipy.misc.lena())))

# seed the number generator, always the same results.
import random



plotIt = True
create_difference_file = True

#print(ourMark)

lengths = [10, 100, 1000, 2000, 4000, 8000, 16000]
for i in lengths:
    random.seed(0)
    print("Current length: %d" % i)
    ourMark = [random.gauss(0,1) for x in range(0,i)]
    cox.simple_embed(input_file="Lenna.bmp",output_file="watermarked.png",watermark=ourMark)
    a = cox.simple_test(orig_file="Lenna.bmp",target_file="watermarked.png",watermark=ourMark)
    print(a)

print("Testing against various, with previous watermarked")
lengths = [10, 50, 100, 150, 200, 400, 800]
for i in lengths:
    random.seed(0)
    print("Current length: %d" % i)
    ourMark = [random.gauss(0,1) for x in range(0,i)]
    a = cox.simple_test(orig_file="Lenna.bmp",target_file="watermarked.png",watermark=ourMark)
    print(a)



if (create_difference_file):
    import watermark.image
    watermark.image.diff_file(input_file1="Lenna.bmp", input_file2="watermarked.png", output_file="difference.png")

