import scipy

"""
    Initially the colorsys module was used, and every rgb tuple was converted
    seperately, however this was found to be too slow. The switch was made to
    matrix vector-notation which allows use of scipy's linear algebra.
"""

# from colorsys:
#y = 0.30*r + 0.59*g + 0.11*b
#i = 0.60*r - 0.28*g - 0.32*b
#q = 0.21*r - 0.52*g + 0.31*b


def rgb_to_yiq_vector(v):
    """
        Function which can convert a vector of (3, N) RGB values to a vector
        of YIQ values.
    """
    return scipy.dot(scipy.array([[0.3, 0.59, 0.11],
                                [0.6, -0.28, -0.32],
                                [0.21, -0.52, 0.31]]), v)

#r = y + 0.948262*i + 0.624013*q
#g = y - 0.276066*i - 0.639810*q
#b = y - 1.105450*i + 1.729860*q
# with  0<r<1, 0<g<1, 0<b<1


def yiq_to_rgb_vector(v, min_val=0, max_val=255.0):
    """
        Function which can convert a vector of (3, N) YIQ values to a vector
        of RGB values.
    """
    x = scipy.dot(scipy.array([[1, 0.948262, 0.624013],
                                [1, -0.276066, -0.639810],
                                [1, -1.105450, 1.729860]]), v)
    x[x > max_val] = max_val # x > max_val returns indices.
    x[x < min_val] = min_val # values at these indices are overwritten.
    return x


def rgb_to_yiq_img(indata):
    """
        Takes argument `indata` which should be a scipy array of shape
        (width, height, 3). So containing the RGB values as last index.

        This function reshapes the RGB data into a vector on which the
        rgb_to_yiq_vector function is then called. This result vector is
        reshaped back to the original dimensions.
    """
    inshape = indata.shape
    rgb = indata.reshape(-1, 3).transpose()
    yiq_data = rgb_to_yiq_vector(rgb)
    yiq_matr = yiq_data.transpose().reshape(inshape[0], inshape[1], 3)
    return yiq_matr


def yiq_to_rgb_img(indata):
    """
        Takes argument `indata` which should be a scipy array of shape
        (width, height, 3). So containing the YIQ values as last index.

        This function reshapes the YIQ data into a vector on which the
        yiq_to_rgb_vector function is then called. This result vector is
        reshaped back to the original dimensions.
    """
    inshape = indata.shape
    yiq = indata.reshape(-1, 3).transpose()
    rgb_data = yiq_to_rgb_vector(yiq)
    rgb_matr = rgb_data.transpose().reshape(inshape[0], inshape[1], 3)
    return rgb_matr


def debug_plot(a):
    # This is just a debugging function....
    import matplotlib.pyplot as plt
    import matplotlib
    # scale it.
    #scipy.min(a, axis=None), scipy.max(a, axis=None)
    #b = ((a - a.min()) / (a.max() - a.min())) * 255
    b = a
    plt.imshow(b, cmap=matplotlib.cm.Greys_r)
    plt.show()
