#!/usr/bin/env python2

import scipy
import scipy.fftpack
import scipy.signal
from math import sqrt
from util import rgb_to_yiq_matrix,yiq_to_rgb_matrix
from util import rgb_to_yiq_img,yiq_to_rgb_img

from util import debug_plot


def embed_file(inputfile, outputfile, watermark, alpha=0.1):
    """
        :parameters:
            inputfile
                Filename for the input image.
            outputfile
                Filename for the resultant image.
            watermark
                This should be an iterable, and return values which mean 0.
            alpha
                Specifies how strongly the watermark is embedded. Alpha 0.1 is
                the default and from the paper.
    """
    rgb_in = scipy.misc.imread(inputfile)
    out_rgb = embed(rgb_in,watermark,alpha)
    
    outdata = (out_rgb).astype('uint8')#
    #print(outdata.shape)

    # Step 10, write this file.
    scipy.misc.imsave(outputfile, outdata)
    


def test_file(origfile,suspectfile, watermark,alpha=0.1):
    """
    :parameters:
        suspectfile
            Filename for with a suspected watermark.
        origfile
            Filename to the original file without watermark.
        watermark
            Watermark which we are testing.
    """
    orig_rgb = scipy.misc.imread(origfile)
    suspect_rgb = scipy.misc.imread(suspectfile)
    test(orig_rgb,suspect_rgb, watermark,alpha=0.1)



def embed(input_rgb, watermark, alpha=0.1):
    
    # Step 1, convert RGB TO YIQ and acquire Y.
    inshape = input_rgb.shape
    #print("INSHAPE RGB->YIQ:")
    #print(inshape)
    yiq_indata = rgb_to_yiq_img(input_rgb)
    #print(yiq_indata.shape)

    # width x height x Y component matrix.
    y_indata = yiq_indata[:,:,0]


    # Step 2, use the acquired Y data to perform a 2 dimensional DCT.

    # define shorthands.
    dct = lambda x: scipy.fftpack.dct(x, norm='ortho')
    idct = lambda x: scipy.fftpack.idct(x, norm='ortho')

    # Perform the computation.
    in_dct = dct(dct(y_indata).transpose(1,0)).transpose(0,1).transpose(0,1)
    print(in_dct)
    # Step 3, convert these DCT components back to a vector once again.
    in_dctv = in_dct.reshape(1,-1)[0]
    print(in_dctv)

    # Step 4, sort this vector, so we know where the highest components are.
    in_dctv_s = scipy.argsort(in_dctv)
    in_dctv_s = in_dctv_s[::-1]


    # Step 5, embed the sequence in the highest N components.
    out_dctv = in_dctv # from now on its the output
    for i,v in enumerate(watermark):
        # embedding using eq (2) from paper.
        # skipping the DC component as well.
        out_dctv[in_dctv_s[i+1]] = in_dctv[in_dctv_s[i+1]] * (1.0 + v * alpha)

    # that's it. The watermark is now embedded, all that remains is restoring
    # it back to a proper image.
    
    # Step 6, create the DCT matrix again.
    out_dct = out_dctv.reshape(inshape[0],inshape[1])

    # Step 7, perform the inverse of the DCT transform.
    y_outdata = idct(idct(out_dct).transpose(1,0)).transpose(0,1).transpose(0,1)
    
    # Step 8, recompose the Y component with its IQ components.
    yiq_outdata = yiq_indata
    # overwrite th Y component.
    yiq_outdata[:,:,0] = y_outdata
    
    # Step 9, convert our YIQ Nx3 matrix back to RGB of original size.
    outdata = yiq_to_rgb_img(yiq_outdata)
    

    # return it, the embedding process is done.
    return outdata


def test(orig_rgb,suspect_rgb, watermark,alpha=0.1):

    # define shorthands
    dct = lambda x: scipy.fftpack.dct(x, norm='ortho')

    
    yiq_suspectdata = rgb_to_yiq_img(suspect_rgb)
    y_suspectdata = yiq_suspectdata[:,:,0]
    suspect_dct = dct(dct(y_suspectdata).transpose(1,0)).transpose(0,1).transpose(0,1)
    suspect_dctv = suspect_dct.reshape(1,-1)[0]
    suspect_dctv_s = scipy.argsort(suspect_dctv)[::-1]


    yiq_origdata = rgb_to_yiq_img(orig_rgb)
    y_origdata = yiq_origdata[:,:,0]
    orig_dct = dct(dct(y_origdata).transpose(1,0)).transpose(0,1).transpose(0,1)
    orig_dctv = orig_dct.reshape(1,-1)[0]
    orig_dctv_s = scipy.argsort(orig_dctv)[::-1]
    #print(orig_dctv_s[0:5])

    
    Xstar = scipy.zeros((len(watermark)))
    for i,v in enumerate(watermark):
        #print("\n\n index: %d, %f" % (i,v))
        #print("Suspect has: %f at %d" % (suspect_dctv[suspect_dctv_s[i]],suspect_dctv_s[i]))
        #print("Orig has: %f at %d" % (orig_dctv[orig_dctv_s[i]],orig_dctv_s[i]))

        # inverse of eq (2)
        #x =  (suspect_dctv[suspect_dctv_s[i+1]] - orig_dctv[orig_dctv_s[i+1]])/(orig_dctv[orig_dctv_s[i+1]]*alpha)
        x =  (suspect_dctv[orig_dctv_s[i+1]] - orig_dctv[orig_dctv_s[i+1]])/(orig_dctv[orig_dctv_s[i+1]]*alpha)
        Xstar[i] = x
        #Xstar[i] = -1 if x < 0 else 1

    #print(Xstar)
    print(' '.join([str("%+1.1f" % x) for x in Xstar[0:20]]))
    
    import random
    random.seed(89198189119189)

    #ourMark = [int(random.choice((0,1))) for x in range(1,ourLength+1)]
    others = len(watermark)
    marks = scipy.zeros((others,len(watermark)))
    for i in range(0,others):
        if (i == round(others/2)):
            #marks[i,:] = [ -1 if x < 0 else 1 for x in watermark]
            marks[i,:] = watermark
            continue;
        marks[i,:] = [random.gauss(0,1) for x in range(0,len(watermark))]
        #marks[i,:] = [int(random.choice((-1,1))) for x in range(0,len(watermark))]
    
    score = scipy.zeros((others,1))
    for i in range(0,others):
        score[i] = scipy.dot(Xstar,marks[i]) / sqrt(scipy.dot(Xstar,Xstar))

    #print(score)
    print("Score of hit: %f" % score[round(others/2)])
    print("max, min, mean: %f, %f, %f" % (max(score), min(score), scipy.mean(score)))

    import matplotlib.pyplot as plt
    import matplotlib
    plt.plot(range(0,others),score,'k')
    plt.plot(round(others/2),score[round(others/2)], 'r*',markersize=30)
    #plt.plot(range(0,len(watermark)), (suspect_dctv[suspect_dctv_s[1:len(watermark)+1]] - orig_dctv[orig_dctv_s[1:len(watermark)+1]])/(orig_dctv[orig_dctv_s[1:len(watermark)+1]]*alpha))
    #plt.plot(range(0,len(watermark)), scipy.array(watermark)*alpha*3 ,'g')
    plt.show()
    



# very useful during debugging:
#import code
#code.interact(local=locals())