
import scipy
import scipy.fftpack
import scipy.signal

import random
from math import sqrt
from .util import rgb_to_yiq_img, yiq_to_rgb_img

def embed_function(v, o, alpha=0.1):
    #Vapos = im.dct_o(i+1) * (1.0 + v * alpha)
    return o * (1.0 + alpha * v)

def extract_function(V, Vstar, alpha=0.1):
    # inverse of the embed function...
    return (Vstar - V) / (V * alpha)

def random_wm_function(length=1000, mu=0, sigma=1):
    """
        Function which returns a random watermark.
    """
    return [random.gauss(mu, sigma) for x in range(0, length)]


# cox' paper is always in DCT...
class Tester(object):
    """
        This is the tester class for the Discrete Cosine Transform watermarking
        as described in the paper by Cox et al.
    """
    def __init__(self, target, original, alpha=0.1, length=1000):
        """
            Initialise the testing object, with the various parameters.

            :parameters:
                target
                    The target file that is being worked on.
                    Can be an object of type `YIQ_DCT_Image` or it can be a
                    string to a pathname, in which a `YIQ_DCT_Image` object is 
                    constructed using this file.
                original
                    The original file as used in the embedding procedure.
                    Can be an object of type `YIQ_DCT_Image` or it can be a
                    string to a pathname, in which a `YIQ_DCT_Image` object is 
                    constructed using this file.
                alpha
                    This is the alpha parameter, in the paper it determines how
                    strongly the watermark is embedded, it is passed unchanged 
                    to the embed and extract function.

            The functions `extract_function` and `random_wm_function` can be
            replaced by the user and are called as follows::
                
                extract_function(Vstar=Vstar, V=V, alpha=self.alpha)
                random_wm_function(length=length)
                
        """
        if (type(target) == str):
            # create the image.
            target = YIQ_DCT_Image.open(target)
        self.target = target

        if (type(original) == str):
            original = YIQ_DCT_Image.open(original)
        self.original = original
        
        self.target.set_dct_indices(self.original.get_dct_indices())

        self.extract_function = extract_function
        #self.embed_function = embed_function
        self.random_wm_function = random_wm_function

        self.alpha = alpha
        self.Xstar = None
        self.sigma = None

    def extract(self, length):
        """
            Function for extracting the embedded values using the original and
            the target image.

            :parameters:
                length
                    Number of coefficients to extract.

            Returns a list containing the extracted values.
        """
        Xstar = scipy.zeros((length))

        for i in range(0, length):
            # inverse of eq (2)
            Vstar = self.target.old(i + 1)
            V = self.original.old(i + 1)

            x = self.extract_function(Vstar=Vstar, V=V, alpha=self.alpha)
            Xstar[i] = x
        self.Xstar = Xstar
        self.XstarRS = sqrt(scipy.dot(self.Xstar, self.Xstar))# root square
        return list(Xstar)

    def response(self, length, N=1000):
        """
            Calculates the random response of N random watermarks against the
            extracted watermark.
            :parameters:
                length
                    Length of the watermark to compare against. 
                N
                    Number of random watermarks to check against.
        """
        score = [0 for i in range(0, N)]
        for i in range(0, N):
            score.append(scipy.dot(self.Xstar,
                    self.random_wm_function(length=length)) / self.XstarRS)
        self.sigma = scipy.std(score)
        return self.sigma, score

    def test(self, watermark, N=1000, threshold=6):
        """
            Test a current watermark against a set of random watermarks.
            :parameters:
                watermark
                    The watermark to check. The length of this watermark is
                    used for comparison.
                N
                    Number of random watermarks to check against.
                threshold
                    The minimum factor by which the test result should exceed
                    the standard deviation.

            Returns a tuple (testresult, (sigma, score)).
        """

        if ((self.Xstar == None) or (len(self.Xstar) != len(watermark))):
            self.extract(length=len(watermark))

        # Step 4: create n random watermarks.
        if (self.sigma == None):
            self.response(N=N, length=len(watermark))

        suspectscore = scipy.dot(self.Xstar, watermark) / self.XstarRS
        testresult = suspectscore > threshold * self.sigma
        return testresult, (self.sigma, suspectscore)




class Marker(object):
    def __init__(self,target, original=None, alpha=0.1):
        """
            Initialise the testing object, with the various parameters.

            :parameters:
                target
                    The target file that is being worked on.
                    Can be an object of type `YIQ_DCT_Image` or it can be a
                    string to a pathname, in which a `YIQ_DCT_Image` object is 
                    constructed using this file.
                original
                    The original file to be used as original in the embedding
                    procedure. When it is not provided it is taken the same as
                    the target file.
                    Can be an object of type `YIQ_DCT_Image` or it can be a
                    string to a pathname, in which a `YIQ_DCT_Image` object is 
                    constructed using this file.
                alpha
                    This is the alpha parameter, in the paper it determines how
                    strongly the watermark is embedded, it is passed unchanged 
                    to the embed and extract function.

            The functions `embed_function` and `random_wm_function` can be
            replaced by the user and are called as follows::
                
                embed_function(v=v, alpha=self.alpha,
                                                o=self.original.old(i + 1))
                random_wm_function(length=length)
        """
        if (type(target) == str):
            # create the image.
            target = YIQ_DCT_Image.open(target)
        self.target = target

        if (type(original) == str):
            original = YIQ_DCT_Image.open(original)

        if (original != None):
            self.original = target
            self.target.set_dct_indices(self.original.get_dct_indices())
        else:
            self.original = target

        self.extract_function = extract_function
        self.embed_function = embed_function
        self.random_wm_function = random_wm_function

        self.alpha = alpha

    def embed(self,watermark):
        """
            Embed the provided watermark value into the current target.
            :parameters:
                watermark
                    This should be an iterable of which the values will be
                    provided to the embed function as value `v`.
                    With the default, this should be some numeric. (Usually
                    from the standard normal distribution.)
        """
        for i, v in enumerate(watermark):
            # embedding using eq (2) from paper.
            # Skipping the DC component during embedding just as in the paper.
            self.target.new(i + 1, self.embed_function(v=v,alpha=self.alpha,
                                            o=self.original.old(i + 1)))
    def output(self):
        """
            Return the current target, which was manipulated.
        """
        return self.target



class YIQ_DCT_Image(object):
    """
        This class allows for easy manipulation of the DCT coefficients.

        :parameters:
            input_rgb
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
        self.y_dct_sn = in_dctv_s.copy()  # idem
        self.y_dct = in_dctv       # vector of values, non-sorted
        self.y_dct_n = in_dctv.copy()     # idem

    @classmethod
    def open(cls, path):
        """
            Construct this object from a file, as specified by `path`.
        """
        return cls(scipy.misc.imread(path).astype('f'))

    def write(self, path):
        """
            Write RGB data to file specified by path.
        """
        scipy.misc.imsave(path, self.rgb())

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

    def old(self, i):
        """
            Get the original value which is represented by the i'th index in
            the current sorted indices.
        """
        return self.y_dct[self.y_dct_sn[i]]

    # set the value. in the new entry.
    def new(self, i, v):
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

    def pixel_count(self):
        """
            Returns the number of pixels in the image.
        """
        return self.inshape[0] * self.inshape[1]


def simple_embed(input_file, output_file, watermark):
    #input_file = YIQ_DCT_Image(scipy.misc.imread(input_file).astype('f'))
    mark = Marker(input_file)
    mark.embed(watermark)
    scipy.misc.imsave(output_file, mark.output().rgb())


def simple_test(orig_file, target_file, watermark):
    #orig_file = YIQ_DCT_Image(scipy.misc.imread(orig_file).astype('f'))
    #target_file = YIQ_DCT_Image(scipy.misc.imread(target_file).astype('f'))
    tester = Tester(original=orig_file, target=target_file)
    
    return tester.test(watermark)
