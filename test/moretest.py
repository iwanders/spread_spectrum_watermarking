#!/usr/bin/env python2

# import the module
try:
    from watermark import cox
except ImportError:
    print("Module not installed. Using one in ../")
    import sys
    sys.path.insert(0, "..")
    from watermark import cox





ourLength = 1000
alpha = 0.1 # 0.1 is default
plotIt = True



# seed the number generator, always the same results.
import random
random.seed(0)

watermarks = []

# create a watermark and add it to the list.
ourMark = [random.gauss(0,1) for x in range(0,ourLength)]
watermarks.append(ourMark)
print("Embedding watermark[0] into watermarked0.bmp")
cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked0.bmp",watermark=ourMark,alpha=alpha)

ourMark = [random.gauss(0,1) for x in range(0,ourLength)]
watermarks.append(ourMark)
print("Embedding watermark[1] into watermarked1.bmp")
cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked1.bmp",watermark=ourMark,alpha=alpha)

# cause some damage to the watermark... (uses the 'new' highest DCT coeffs)
# This is a HIGHLY directed attack, Watermark 1 should still be present.
scrambleMark = [random.gauss(0,1) for x in range(0,ourLength)]
print("Attacking watermarked1.bmp with a bogus mark. Watermark[1] should still be detectable in watermarked2.bmp.")
cox.embed_file(inputfile="watermarked1.bmp",outputfile="watermarked2.bmp",watermark=scrambleMark,alpha=alpha)



# walk over watermarks:
for i in range(0,2):
    # walk over images.
    for j in range(0,3):
        filename = "watermarked%d.bmp"%j
        a = cox.test_file(origfile="Lenna.bmp",suspectfile=filename,watermark=watermarks[i],alpha=alpha)
        print("Watermark %d present in %s  -> %s (std: %f, score: %f)" % (i,filename,a["test"], a["stats"][0],a["stats"][1]))
        if (plotIt):
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("Should have matplotlib for plotting.")
            plt.figure()
            plt.plot(range(0,len(a["scores"])), a["scores"],'k')
            # mark the suspected watermark with a red circle.
            plt.plot(a["index"], a["scores"][a["index"]],'o r',markersize=10)
            # show plot.
            #plt.show()
            plt.savefig("test_wm%d_file%d.png"%(i,j), transparent=True)
