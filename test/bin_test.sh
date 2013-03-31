#!/bin/bash

echo "-> This runs several dull tests.."

#../wmark.py help embed
#../wmark.py help test
echo -e "\n"
echo "-> Creating new watermark in Lenna_wm_bin.wm and watermarking Lenna.bmp
using the default output filename. Then doing this with a specified output file
which is identical the md5 hashes should be identical."

../wmark.py embed -c -f -m "Lenna_wm_bin1.wm" Lenna.bmp
md5sum Lenna_wm.bmp
../wmark.py embed -f -m "Lenna_wm_bin1.wm" -o "Lenna_wm.bmp" Lenna.bmp
md5sum Lenna_wm.bmp
echo -e "\n"


echo -e "\n"
echo "-> Next, the Lenna_wm.bmp file is tested against the original watermark."
../wmark.py test -m "Lenna_wm_bin1.wm"

echo "-> We create a second watermarked image, just to acquire a watermark"
../wmark.py embed -f -c -m "Lenna_wm_bin2.wm" -o "Lenna_wm2.bmp" Lenna.bmp

echo "-> Testing the first watermarked image against the second created water-
mark should fail:"
../wmark.py test -m "Lenna_wm_bin2.wm" Lenna_wm.bmp

echo "-> Testing the first 100 entries of the watermark only."
../wmark.py test -l 100 -m "Lenna_wm_bin1.wm"

#echo "->Test the first watermarked image using the second as a base file... this is utter nonsense..."
#../wmark.py test -m "Lenna_wm_bin1.wm" -b "Lenna_wm2.bmp" Lenna_wm.bmp
echo -e "\n"
echo -e "-> Last Test:
        Recreate the second watermarked im using the first watermarked image as a
        base, save as Lenna_wm2.bmp.\n"
../wmark.py embed -f -m "Lenna_wm_bin2.wm" -b Lenna_wm.bmp -o Lenna_wm2.bmp Lenna_wm.bmp
echo "-> Check Lenna_wm2.bmp against Lenna_wm.bmp with the second watermark"
../wmark.py test -m "Lenna_wm_bin2.wm" -b "Lenna_wm.bmp" Lenna_wm2.bmp
echo "-> Check Lenna_wm2.bmp against Lenna.bmp with the first watermark"
../wmark.py test -m "Lenna_wm_bin1.wm" -b "Lenna.bmp" Lenna_wm2.bmp

echo -e "\n"

echo "In the last test the following was done.:
    Watermark 1 (WM1) was created &
        : Embedded in Lenna.bmp -> Lenna_wm.bmp
        : Testing against Lenna.bmp gave positive result. (as we embedded this)
    Watermark 2 (WM2) was created &
        : Embedded in Lenna_wm.bmp (with Lenna_wm.bmp) -> Lenna_wm2.bmp

    Testing WM2 in Lenna_wm2 using Lenna_wm as base is positive.
    Testing WM1 in Lenna_wm2 using Lenna as base is also positive. So a chain
    has been created of watermark embedding.

    One could also see this last thing as an attack on the first watermark. As
    a lot of the coefficients used will be the same."