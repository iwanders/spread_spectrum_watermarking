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


import random
import scipy
import scipy.ndimage


plotIt = True
create_difference_file = True

#print(ourMark)

#input_path="Lenna.bmp"

lengths = [100, 1000, 2000, 4000, 8000]
for i in lengths:
    random.seed(0)
    print("Current length: %d" % i)
    ourMark = [random.gauss(0,1) for x in range(0,i)]
    input_image = cox.YIQ_DCT_Image.open(input_path)
    mark = cox.Marker(input_image)
    mark.embed(ourMark)
    watermarkedData = mark.output().rgb()
    shape = watermarkedData.shape

    for i in [0.90, 0.75,0.5, 0.25, 0.125, 0.1]:
        #blurreddata = scipy.ndimage.filters.gaussian_filter(watermarkedData, sigma=(i, i, 0),) # blurring in the depth direction is an aweful bad idea...
        resizeddata = scipy.misc.imresize(watermarkedData, i) 
        backsizeddata = scipy.misc.imresize(resizeddata, shape )
        cox.YIQ_DCT_Image(backsizeddata).write('resized.png')

        target_image = cox.YIQ_DCT_Image(backsizeddata)
        input_image = cox.YIQ_DCT_Image.open(input_path)
        tester = cox.Tester(target=target_image,original=input_image)
        res = tester.test(ourMark)
        if (not res[0]):
            for j in [0.9, 0.75, 0.5, 0.4, 0.3, 0.2]:
                res = tester.test(ourMark[0:int(j*len(ourMark))])
                if (res[0]):
                    print("Succes with a reduction of %f, new length: %d : %s " %(j,int(j*len(ourMark)),res))
        print("Testing resize blur %f: %s " % (i,res))
