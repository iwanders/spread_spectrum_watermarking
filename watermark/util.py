import scipy

def rgb_to_yiq_matrix(a):
    # from colorsys:
    #
    #def rgb_to_yiq(r, g, b):
    #    y = 0.30*r + 0.59*g + 0.11*b
    #    i = 0.60*r - 0.28*g - 0.32*b
    #    q = 0.21*r - 0.52*g + 0.31*b
    #    return (y, i, q)
    # but element wise handling is slow ->
    # use Numpy's matrix calculation backend.
    return scipy.dot(scipy.array([[0.3, 0.59, 0.11],
                                [0.6, -0.28, -0.32],
                                [0.21, -0.52, 0.31]]),a.transpose())



def yiq_to_rgb_matrix(a):
    #r = y + 0.948262*i + 0.624013*q
    #g = y - 0.276066*i - 0.639810*q
    #b = y - 1.105450*i + 1.729860*q
    # with  0<r<1, 0<g<1, 0<b<1
    x = scipy.dot(scipy.array([[1, 0.948262, 0.624013],
                                [1, -0.276066, -0.639810],
                                [1, -1.105450, 1.729860]]),a)
    # still have to enforce the limits... method sucks, works for now.
    min_val = 0
    max_val = 255
    for i in range(0,x.shape[1]):
        x[0,i] = min(max(x[0,i],min_val),max_val)
        x[1,i] = min(max(x[1,i],min_val),max_val)
        x[2,i] = min(max(x[2,i],min_val),max_val)
    return x






def debug_plot(a):
    import matplotlib.pyplot as plt
    import matplotlib
    # scale it.
    #scipy.min(a, axis=None), scipy.max(a, axis=None)
    b = ((a - a.min()) / (a.max() - a.min())) * 255
    plt.imshow(b,cmap = matplotlib.cm.Greys_r)
    plt.show()