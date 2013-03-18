
import scipy
import scipy.fftpack
import scipy.signal

import random
from math import sqrt
from .util import rgb_to_yiq_img, yiq_to_rgb_img


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
    out_rgb = embed(rgb_in, watermark, alpha)

    outdata = (out_rgb).astype('uint8')
    # write the output.
    scipy.misc.imsave(outputfile, outdata)


def test_file(origfile, suspectfile, watermark, alpha=0.1):
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
    return test(orig_rgb, suspect_rgb, watermark, alpha=0.1)


def embed(input_rgb, watermark, alpha=0.1):
    """
    :parameters:
        input_rgb
            A SciPy array of shape (width,height,3) containing RGB values.
        watermark
            An iterable returning numeric values to embed.
        alpha
            Specifies how strongly the watermark is embedded. Alpha 0.1 is
            the default and from the paper.
    """


    # Step 1, convert RGB TO YIQ and acquire Y.
    inshape = input_rgb.shape

    yiq_indata = rgb_to_yiq_img(input_rgb)
    #print(yiq_indata.shape)

    # width x height x Y component matrix.
    y_indata = yiq_indata[:, :, 0] # cannot be made pep8 compatible.


    # Step 2, use the acquired Y data to perform a 2 dimensional DCT.

    # define shorthands.
    dct = lambda x: scipy.fftpack.dct(x, norm='ortho')
    idct = lambda x: scipy.fftpack.idct(x, norm='ortho')

    # Perform the computation.
    in_dct = dct(dct(y_indata).transpose(1, 0)).transpose(0,
                                                            1).transpose(1, 0)

    # Step 3, convert these DCT components back to a vector once again.
    in_dctv = in_dct.reshape(1, -1)[0]


    # Step 4, sort this vector, so we know where the highest components are.
    in_dctv_s = scipy.argsort(in_dctv)
    in_dctv_s = in_dctv_s[::-1]


    # Step 5, embed the sequence in the highest N components.
    out_dctv = in_dctv # from now on its the output
    for i, v in enumerate(watermark):
        # embedding using eq (2) from paper. Skipping the DC component during
        # embedding just as in the paper.
        Vapos = in_dctv[in_dctv_s[i + 1]] * (1.0 + v * alpha)
        out_dctv[in_dctv_s[i + 1]] = Vapos

    # that's it. The watermark is now embedded, all that remains is restoring
    # it back to a proper image.

    # Step 6, create the DCT matrix again.
    out_dct = out_dctv.reshape(inshape[0], inshape[1])

    # Step 7, perform the inverse of the DCT transform.
    y_outdata = idct(idct(out_dct).transpose(1, 0)).transpose(0,
                                                            1).transpose(1, 0)

    # Step 8, recompose the Y component with its IQ components.
    yiq_outdata = yiq_indata
    # overwrite th Y component.
    yiq_outdata[:, :, 0] = y_outdata

    # Step 9, convert our YIQ Nx3 matrix back to RGB of original size.
    outdata = yiq_to_rgb_img(yiq_outdata)


    # return it, the embedding process is done.
    return outdata


def test(orig_rgb, suspect_rgb, watermark, alpha=0.1, threshold=6):
    """
    :parameters:
        orig_rgb
            A SciPy array of shape (width,height,3) containing RGB values. This
            should be the original image.
        suspect_rgb
            A SciPy array of shape (width,height,3) containing RGB values. This
            is the image we suspect a watermark to be present.
        watermark
            The suspected watermark to test.
        alpha
            Specifies how strongly the watermark is embedded. Alpha 0.1 is
            the default and from the paper.
        threshold
            threshold, number of standard deviations the suspected watermark
            should be away from the standard deviation. T=6 is from the paper.

    :rtype: dictionary with the following entries:
        * test : Boolean indicating whether the response exceeded the
            threshold.
        * stats : Tuple of (standard deviation, score)
        * scores : a list containing the scores of all watermarks against which
            the testing was done.
        * index : integer index where the suspect watermark was placed
            in the scores return value.
    """

    # define shorthand
    dct = lambda x: scipy.fftpack.dct(x, norm='ortho')

    # Step 1: convert the suspect to a Y vector.
    yiq_suspectdata = rgb_to_yiq_img(suspect_rgb)
    y_suspectdata = yiq_suspectdata[:, :, 0]
    suspect_dct = dct(dct(y_suspectdata).transpose(1, 0)).transpose(0,
                                                            1).transpose(0, 1)
    suspect_dctv = suspect_dct.reshape(1, -1)[0]
    suspect_dctv_s = scipy.argsort(suspect_dctv)[::-1] # in descending order

    # Step 2: same for the original
    yiq_origdata = rgb_to_yiq_img(orig_rgb)
    y_origdata = yiq_origdata[:, :, 0]
    orig_dct = dct(dct(y_origdata).transpose(1, 0)).transpose(0,
                                                            1).transpose(0, 1)
    orig_dctv = orig_dct.reshape(1, -1)[0]
    orig_dctv_s = scipy.argsort(orig_dctv)[::-1]

    # Step 3: obtain X* from our suspect
    Xstar = scipy.zeros((len(watermark)))
    for i, v in enumerate(watermark):
        # inverse of eq (2)
        Vstar = suspect_dctv[orig_dctv_s[i + 1]]
        # the paper is not clear if the sorted indices of the suspect should be
        # used here. If they are used however, no watermark is detected.

        # The paper states this; "We extract X* by first extracting a set of
        # values V* = v_1*,...v_n* from D* (using information about D) and then
        # generating X* from V* and V."
        # So it seems reasonable to assume that the original N highest
        # components are used.

        V = orig_dctv[orig_dctv_s[i + 1]]
        x = (Vstar - V) / (V * alpha)
        Xstar[i] = x

    # Step 4: create n random watermarks.
    n = len(watermark)
    marks = scipy.zeros((n, len(watermark)))
    suspectindex = int(round(n / 2))
    for i in range(0, n):

        # if at suspect index (center of results), insert the suspected Xstar.
        if (i == suspectindex):
            marks[i, :] = watermark
            continue

        # picking from N(0,1) with normal(mu,sigma) as in the paper.
        marks[i, :] = [random.gauss(0, 1) for x in range(0, len(watermark))]

    # Step 5: Compute the similarity according to the paper.
    score = [0 for i in range(0, n)]
    for i in range(0,n):
        # sim(mark,Xstar) = (Xstar dotproduct mark)/sqrt(Xstar dotprod Xstar)
        score[i] = scipy.dot(Xstar,marks[i]) / sqrt(scipy.dot(Xstar,Xstar))

    sigma = scipy.std(score)
    suspectscore = score[suspectindex]
    testresult = suspectscore > threshold
    return {"test": testresult,
            "stats": (sigma, suspectscore),
            "scores": score,
            "index": suspectindex}
