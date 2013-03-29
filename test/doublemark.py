#!/usr/bin/env python2

# import the module
try:
    from watermark import cox
except ImportError:
    print("Module not installed. Using one in ../")
    import sys
    sys.path.insert(0, "..")
    from watermark import cox


# seed the number generator, always the same results.
import random
random.seed(0)

ourLength = 2000
ourMark1 = [random.gauss(0,1) for x in range(0,ourLength)]
ourMark2 = [random.gauss(0,1) for x in range(0,ourLength)]

plotIt = True
create_difference_file = True

#print(ourMark)

# Simple test, embed two watermarks and check results


input_path="Lenna.bmp"
output_file="watermarked.png"
input_file = cox.yiq_dct_image.open(input_path)
mark = cox.dctwatermarker(input_file)
mark.wm(ourMark1)
mark.embed()
mark.wm(ourMark2)
mark.embed()
mark.output().write(output_file)



res, stats = cox.simple_test(orig_file="Lenna.bmp",target_file="watermarked.png",watermark=ourMark1)
print("res: %s, stats: %s" % (res, stats))
a = cox.simple_test(orig_file="Lenna.bmp",target_file="watermarked.png",watermark=ourMark2)
print(a)
#print("Watermark present: %s" % a["test"])




# Embed 50 watermarks and test against those.


input_file = cox.yiq_dct_image.open(input_path)
mark = cox.dctwatermarker(input_file)
newwm = []

embedcount = 10
extratestCount = 100

for i in range(0,embedcount):
    newwm.append([random.gauss(0,1) for x in range(0,ourLength)])
    mark.embed_wm(newwm[i])
res = mark.output()
res.write(output_file)


target_image = cox.yiq_dct_image.open(output_file)
print("new obj")
tester = cox.dctwatermarker(target_image)
tester.orig_file(input_path)


for i in range(0,embedcount + extratestCount):
    if (i < len(newwm)):
        print(tester.test_wm(newwm[i]))
    else:
        print(tester.test_wm([random.gauss(0,1) for x in range(0,ourLength)]))



if (create_difference_file):
    import watermark.image
    watermark.image.diff_file(input_file1="Lenna.bmp", input_file2="watermarked.png", output_file="difference.png")

