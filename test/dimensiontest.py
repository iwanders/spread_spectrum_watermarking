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
if (not os.path.lexists('LennaPart.bmp')):
    print("LennaPart does not exist, making it.")
    import scipy
    # save Lenna.bmp as color image (even though it is BW)
    LennaPart = scipy.misc.lena()[30:413,160:376]
    scipy.misc.imsave("LennaPart.png", scipy.dstack((LennaPart,
                                                LennaPart,
                                                    LennaPart)))

# seed the number generator, always the same results.
import random
random.seed(0)

ourLength = 1000
alpha = 0.1 # 0.1 is default
ourMark = [random.gauss(0,1) for x in range(0,ourLength)]

plotIt = True



cox.embed_file(inputfile="LennaPart.png",outputfile="halfwatermarked.png",watermark=ourMark,alpha=alpha)


a = cox.test_file(origfile="LennaPart.png",suspectfile="halfwatermarked.png",watermark=ourMark,alpha=alpha)
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
