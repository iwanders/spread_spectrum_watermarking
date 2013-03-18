#!/usr/bin/env python2

# import the module
try:
    from watermark import cox
except ImportError:
    print("Module not installed. Using one in ../")
    import sys
    sys.path.insert(0, "..")
    from watermark import cox


watermarks = []

# seed the number generator, always the same results.
import random
random.seed(0)

ourLength = 1000
alpha = 0.1 # 0.1 is default


# create a watermark and add it to the list.
ourMark = [random.gauss(0,1) for x in range(0,ourLength)]
watermarks.append(ourMark)

cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked0.bmp",watermark=ourMark,alpha=alpha)

ourMark = [random.gauss(0,1) for x in range(0,ourLength)]
watermarks.append(ourMark)
cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked1.bmp",watermark=ourMark,alpha=alpha)



# walk over watermarks:
for i in range(0,2):
    # walk over images.
    for j in range(0,2):
        filename = "watermarked%d.bmp"%j
        a = cox.test_file(origfile="Lenna.bmp",suspectfile=filename,watermark=watermarks[i],alpha=alpha)
        print("Watermark %d present in %s  -> %s (std: %f, score: %f)" % (i,filename,a["test"], a["stats"][0],a["stats"][1]))