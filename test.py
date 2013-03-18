#!/usr/bin/env python2

from watermark import cox


import random
random.seed(0)

ourLength = 1000
alpha = 0.1 # 0.1 is default
#ourMark = [int(random.choice((-1,1))) for x in range(0,ourLength)]
ourMark = [random.gauss(0,1) for x in range(0,ourLength)]


#ourMark = m
    



#print("Our watermark is below this:")
#print(' '.join([str("%+1.1f" % x) for x in ourMark[0:20]]))


#cox.embed(inputfile="Lenna.bmp",outputfile="watermarked.bmp",watermark=ourMark,alpha=alpha)
cox.embed_file(inputfile="Lenna.bmp",outputfile="watermarked.bmp",watermark=ourMark,alpha=alpha)


#random.seed(564151561)

#ourLength = 1024
#ourMark = [int(random.choice((0,1))) for x in range(0,ourLength)]

#cox.test(origfile="Lenna.bmp",suspectfile="watermarked.bmp",watermark=ourMark,alpha=alpha)




#ourMark = [(0 if (x < 0.0) else 1) for x in ourMark]
#print(ourMark)
