#!/usr/bin/env python2

"""
    This file behaves like an executable for watermarking files.
"""


"""
    To provide various options, call it from another file like::

        if __name__ == "__main__":
            import watermark.bin
            watermark.bin.prefixWatermark([-1, 1])
            watermark.bin.run()
"""


# this file is not pep8 conform.


import argparse
import os.path
import sys


def confirm(text,printMe=True):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
    if (printMe):
        sys.stdout.write(text+": ")
    try:
        choice = raw_input().lower()
    except KeyboardInterrupt:
        return None

    if choice in yes:
       return True
    elif choice in no:
       return False
    else:
        return confirm(text,printMe=False)


# sub-command functions
def embed(args,**kwargs):
    from cox import Marker, YIQ_DCT_Image, random_wm_function
    print(args)
        
    #mark.embed(watermark)
    #scipy.misc.imsave(output_file, mark.output().rgb())
    for f in args.file:
        # f is now the path to the file.
        
        
        (name, ext) = os.path.splitext(f)
        fileProps = {"name":name, "ext":ext}

        target_image = YIQ_DCT_Image.open(f)
        mark = Marker(target=target_image,original=args.base,alpha=args.alpha)
        # now we have to obtain a watermark, one way or another.
        if (args.create):
            # whoa, we should make a new watermark. How cool is that.

            # parse the length command.
            if (args.length == -1):
                length = int(0.0038 * target_image.pixel_count())
            elif (args.length.endswith("%")):
                length = int(0.01* float(args.length[0:-1]) * target_image.pixel_count())
            else:
                length = int(args.length)
            print(length)
            # assume that it is a random command from normal distr.
            
            wm = random_wm_function(length=length)
            print(len(wm))
            
        else:
            # watermark file should be given.
            raise NotImplementedError("Watermark file not implemented")

        mark.embed(watermark=wm)

        result = mark.output()

        # check if the output file is good...
        output_path = args.out.format(**fileProps)
        #print(output_path)
        if (os.path.lexists(output_path)):
            if ((args.force) or confirm('Are you sure you want to overwrite {0}'.format(output_path))):
                print("Overwriting...")
                result.write(output_path)
        else:
            result.write(output_path)

        
            

#    raise NotImplementedError("Embed is not yet implemented.")

def test(args,**kwargs):
    raise NotImplementedError("Test is not yet implemented.")

def help(args,**kwargs):
    subparsers.choices[args.command].print_help()

d = """This command can be used to watermark an image with a digital watermark.
This watermark will be embedded in the discrete cosine transform-domain."""
parser = argparse.ArgumentParser(description=d)

subparsers = parser.add_subparsers(title='subcommands',
                                        description='Valid subcommands',
                                        help='additional help')

# create the parser for the "embed" command
parser_embed = subparsers.add_parser('embed', help="Used to embed a watermark")

# embed related arguments
#parser_embed.add_argument("-t", "--type", help="What type of watermark")
parser_embed.add_argument("-a", "--alpha", help="Alpha, strength of watermark, as ratio, default 0.1.", default=0.1, type=float)
parser_embed.add_argument("-l", "--length", help="Length of watermark, defaults to 0.38%% of the number of pixels. Can be specified with a %% in which case a percentage of the number of pixels will be used, or as a number, in which case the length of the watermark will be equal to this number.", default=-1)

#group = parser_embed.add_mutually_exclusive_group()
# We either use a watermark file, or we create one with a default name.
# if we should create one and the mark file is set we write it to the mark file.
parser_embed.add_argument("-m", "--mark", metavar="<Wm File>",
                help="Path to the watermark data. Default {name}_wm.wm.", default="{name}_wm.wm")
parser_embed.add_argument("-c", "--create", action="store_true",
                    help="Create watermark and save to file", default=False)

# General arguments
parser_embed.add_argument("-f", "--force", help="Assume yes to all questions.",
                    action="store_true", default=False)


parser_embed.add_argument("-o", "--out", metavar="<Output File>",
                help="Path to the new output file. Default {name}_wm.{ext}.", default="{name}_wm{ext}")

# file thingy arguments.
#parser_embed.add_argument("-s", "--suffix", help="Suffix for all output files")
#parser_embed.add_argument("-o", "--outdir", help="Output Directory")

parser_embed.add_argument("-b", "--base", help="Path to the original file")
parser_embed.add_argument("file", help="Path to the image file(s).", nargs='+')

parser_embed.set_defaults(func=embed)







# create the parser for the "test" command
parser_test = subparsers.add_parser('test', help="Used to test for a watermark")
parser_test.set_defaults(func=test)
parser_test.add_argument("-t", "--type", help="What type of watermark")
parser_test.add_argument("-m", "--mark", help="Path to the watermark data.")
parser_test.add_argument("-b", "--base", help="Base file for the coefficients")
parser_test.add_argument("file", help="Path to the image file(s).")





# create the parser for the "help" command
parser_help = subparsers.add_parser('help', help="Get help for a command")
parser_help.add_argument("command",choices=subparsers.choices)
parser_help.set_defaults(func=help)





def run(prefix=[]):
    # parse the args and call whatever function was selected
    args = parser.parse_args()
    args.func(args,prefix=prefix)

if __name__ == "__main__":
    run()

