
import scipy
import scipy.fftpack
import scipy.signal

import random
from math import sqrt
from .util import rgb_to_yiq_img, yiq_to_rgb_img


class dctwatermarker(object):
    """
        This is the watermarker object, it takes a target of type
        yiq_dct_image, on which embed or testing operations can be done.

        By default, the indices of this target image are used.
    """

    def __init__(self, target, alpha=0.1, size=1000):
        self.target = target
        self.original = target
        self.alpha = alpha
        self.size = size
        self.Xstar = None
        self.sigma = None

    def embed_function(self, v, o, alpha=None):
        if (alpha == None):
            alpha = self.alpha
        #Vapos = im.dct_o(i+1) * (1.0 + v * alpha)
        return o * (1.0 + alpha * v)

    def extract_function(self, V, Vstar, alpha=None):
        if (alpha == None):
            alpha = self.alpha
        return (Vstar - V) / (V * alpha)

    def orig(self, orig):
        """
            Use this `orig` as new original image, should be of type
            yiq_dct_image. The highest indices of this image will be used
            instead of those which are present in the target image.
        """
        self.original = orig
        self.target.set_dct_indices(self.original.get_dct_indices())

    def wm_random(self, length=None, mu=0, sigma=1):
        """
            Function which returns a random watermark.
        """
        if (length == None):
            length = self.size
        return [random.gauss(mu, sigma) for x in range(0, length)]

    def wm(self, wm):
        """
            Sets the current watermark to test.
        """
        self.size = len(wm)
        self.Xstar = None
        self.watermark = wm

    def embed(self):
        """
            Embed the self.watermark value into the current target.
        """
        for i, v in enumerate(self.watermark):
            # embedding using eq (2) from paper.
            # Skipping the DC component during embedding just as in the paper.
            self.target.n(i + 1, self.embed_function(v=v,
                                            o=self.original.o(i + 1)))

    def output(self):
        """
            Return the current target, which was manipulated.
        """
        return self.target

    def extract(self):
        """
            Function for extracting the embedded values using the original and
            the target image.
        """
        Xstar = scipy.zeros((self.size))
        #print(Xstar.shape)
        for i in range(0, self.size):
            # inverse of eq (2)
            Vstar = self.target.o(i + 1)

            V = self.original.o(i + 1)
            x = self.extract_function(Vstar=Vstar, V=V)
            Xstar[i] = x
        self.Xstar = Xstar
        self.XstarRS = sqrt(scipy.dot(self.Xstar, self.Xstar))# root square
        return Xstar

    def response(self, N=1000):
        """
            Calculates the random response of N random watermarks against the
            extracted watermark.
        """
        score = [0 for i in range(0, N)]
        for i in range(0, N):
            score.append(scipy.dot(self.Xstar, self.wm_random()) /
                                                                self.XstarRS)
        self.sigma = scipy.std(score)
        return self.sigma

    def test(self, N=1000, threshold=6):
        """
            Test the current watermark against a set of random watermarks.
        """

        if (self.Xstar == None):
            self.extract()

        # Step 4: create n random watermarks.
        if (self.sigma == None):
            self.response(N)

        suspectscore = scipy.dot(self.Xstar, self.watermark) / self.XstarRS
        testresult = suspectscore > threshold * self.sigma
        return {"test": testresult, "stats": (self.sigma, suspectscore)}


class yiq_dct_image(object):
    """
        This class allows for easy manipulation of the DCT coefficients.

        :parameters:
            sinput_rgb
                A SciPy array of shape (width,height,3) containing RGB values.

    """

    def __init__(self, input_rgb):
        dct = lambda x: scipy.fftpack.dct(x, norm='ortho')
        self.inshape = input_rgb.shape

        yiq_indata = rgb_to_yiq_img(input_rgb)
        self.yiq = yiq_indata
        #print(yiq_indata.shape)

        # width x height x Y component matrix.
        y_indata = yiq_indata[:, :, 0] # cannot be made pep8 compatible.

        # Step 2, use the acquired Y data to perform a 2 dimensional DCT.

        # define shorthands.

        # Perform the computation.
        in_dct = dct(dct(y_indata).transpose(1, 0)).transpose(0,
                                                            1).transpose(1, 0)

        # Step 3, convert these DCT components back to a vector once again.
        in_dctv = in_dct.reshape(1, -1)[0]


        # Step 4, sort this vector,so we know where the highest components are.
        in_dctv_s = scipy.argsort(in_dctv)
        in_dctv_s = in_dctv_s[::-1]

        self.y_dct_s = in_dctv_s   # vector of indices, sorted by magnitude
        self.y_dct_sn = in_dctv_s  # idem
        self.y_dct = in_dctv       # vector of values, non-sorted
        self.y_dct_n = in_dctv     # idem

    def get_dct_indices(self):
        """
            Allows retrieval of a vector of indices. Sorted by magnitude, in
            descending order of magnitude.
        """
        return self.y_dct_s

    def set_dct_indices(self, indices):
        """
            Allows setting the list of indices to use. For use with the o and n
            functions.
        """
        self.y_dct_sn = indices

    # get the value
    def o(self, i):
        """
            Get the original value which is represented by the i'th index in
            the current sorted indices.
        """
        return self.y_dct[self.y_dct_sn[i]]

    # set the value. in the new entry.
    def n(self, i, v):
        """
            Set the i'th index to value v.
        """
        self.y_dct_n[self.y_dct_sn[i]] = v

    def rgb(self):
        """
            Recompose this object in an an RGB matrix; (width,height,3).
        """
        idct = lambda x: scipy.fftpack.idct(x, norm='ortho')
        # Step 6, create the DCT matrix again.
        out_dct = self.y_dct_n.reshape(self.inshape[0], self.inshape[1])

        # Step 7, perform the inverse of the DCT transform.
        y_outdata = idct(idct(out_dct).transpose(1, 0)).transpose(0,
                                                            1).transpose(1, 0)

        # Step 8, recompose the Y component with its IQ components.
        yiq_outdata = self.yiq
        # overwrite th Y component.
        yiq_outdata[:, :, 0] = y_outdata

        # Step 9, convert our YIQ Nx3 matrix back to RGB of original size.
        outdata = yiq_to_rgb_img(yiq_outdata)
        return outdata


def simple_embed(input_file, output_file, watermark):
    input_file = yiq_dct_image(scipy.misc.imread(input_file).astype('f'))
    mark = dctwatermarker(input_file)
    mark.wm(watermark)
    mark.embed()
    scipy.misc.imsave(output_file, mark.output().rgb())


def simple_test(orig_file, target_file, watermark):
    orig_file = yiq_dct_image(scipy.misc.imread(orig_file).astype('f'))
    target_file = yiq_dct_image(scipy.misc.imread(target_file).astype('f'))
    mark = dctwatermarker(target_file)
    mark.orig(orig_file)
    mark.wm(watermark)
    return mark.test()
