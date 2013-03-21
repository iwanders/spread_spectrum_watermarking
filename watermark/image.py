
import scipy

#import random
#from .util import rgb_to_yiq_img, yiq_to_rgb_img


def diff_file(input_file1, input_file2, output_file):
    """
        :parameters:
            input_file1
                Filename for the input image 1.
            outputfile
                Filename for the input image 2.
            output_file
                Output file name, contains comparison of both.
    """
    rgb_in1 = scipy.misc.imread(input_file1)
    rgb_in2 = scipy.misc.imread(input_file2)
    if (rgb_in1.shape != rgb_in2.shape):
        raise TypeError("Not identical size")

    out_rgb = scipy.ones(rgb_in1.shape) * 128.0 + (rgb_in1 - rgb_in2)
    outdata = (out_rgb).astype('uint8')
    # write the output.
    scipy.misc.imsave(output_file, outdata)
