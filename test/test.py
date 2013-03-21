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
random.seed(0)

ourLength = 1000
alpha = 0.1 # 0.1 is default
ourMark = [random.gauss(0,1) for x in range(0,ourLength)]

plotIt = True
create_difference_file = True



cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked.png",watermark=ourMark,alpha=alpha)


a = cox.test_file(origfile="Lenna.bmp",suspectfile="watermarked.png",watermark=ourMark,alpha=alpha)
print("Watermark present: %s" % a["test"])


if (plotIt):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Should have matplotlib for plotting.")

    # plot all random stuff.
    plt.plot(range(0,len(a["scores"])), a["scores"],'k')
    # mark the suspected watermark with a red circle.
    plt.plot(a["index"], a["scores"][a["index"]],'o r',markersize=10)
    # show plot.
    plt.show()
    #plt.savefig("test_result.png", transparent=True)


if (create_difference_file):
    import watermark.image
    watermark.image.diff_file(input_file1="Lenna.bmp", input_file2="watermarked.png", output_file="difference.png",mode='rgb')

