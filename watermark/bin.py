#!/usr/bin/env python2
__version__ = "0.0.1"


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

default_watermark_file = "{name}_wm.wm"
default_alpha = 0.1
default_length = 1000


def confirm(text,printMe=True,default=True):
    # modified from http://stackoverflow.com/questions/3041986/python-command-line-yes-no-input

    yes = set(['yes','y', 'ye'])
    no = set(['no','n'])
    if (default):
        yes.add('')
    else:
        no.add('')

    # catch errors
    if (printMe):
        sys.stdout.write(text+": ")
    try:
        choice = raw_input().lower()
    except KeyboardInterrupt:
        return False

    if choice in yes:
       return True
    elif choice in no:
       return False
    else:
        sys.stdout.write("A yes/no will suffice:")
        return confirm(text,printMe=False)

def exit(text):
    sys.stdout.write(text + ", exiting\n")
    sys.exit()



def write_watermark(wm,file_wm, file_base=None, file_target=None,alpha=default_alpha, version=__version__, type="JSON"):
    if (type == "JSON"):
        storage = {"wm":file_wm,"target":file_target,"base":file_base,"version":version,"type":type,"alpha":alpha,"wm":wm,"length":len(wm)}
        import json
        with open(file_wm,'w') as f:
            json.dump(storage, f, indent=True,sort_keys=True)
        
def read_watermark(file_wm,type="JSON"):
    if (type == "JSON"):
        import json
        with open(file_wm,'r') as f:
            data = json.load(f)
            if (data["version"] == "0.0.1"): # hardcoded here.. yes :)
                return data


def make_watermark(length,prefix=[]):
    from cox import random_wm_function
    prefixLength = len(prefix)
    desiredlength = length - prefixLength
    watermark = list(prefix)
    randomPart = random_wm_function(length=desiredlength)
    watermark.extend(randomPart)
    return watermark

# sub-command functions
def embed(args,**kwargs):
    from cox import Marker, YIQ_DCT_Image, random_wm_function
    
    
    #mark.embed(watermark)
    #scipy.misc.imsave(output_file, mark.output().rgb())
    for f in args.file:
        # f is now the path to the file.
        # TODO: args.base checking...
        
        (name, ext) = os.path.splitext(f)
        fileProps = {"name":name, "ext":ext}
        mark_path = args.mark.format(**fileProps)
        output_path = args.out.format(**fileProps)
        base_path = args.base if (args.base != None) else f
        alpha = float(args.alpha)

        # now we have to obtain a watermark, one way or another.
    
        # parse the length command.
        if (str(args.length).endswith("%")):
            length = int(0.01* float(args.length[0:-1]) * target_image.pixel_count())
        else:
            length = int(args.length)

        if (args.createnew):
             # whoa, we should make a new watermark. How cool is that.
            # assume that it is a random command from normal distr.
            wm = make_watermark(length,prefix=kwargs["prefix"])

        elif (os.path.lexists(mark_path)):# if we shouldn't make a new one, try to get the current watermark from file.
            print("Watermark file found, reading.")
            wmdata = read_watermark(mark_path,type="JSON")
            wm = wmdata["wm"]
        else: # that watermark file didn't exist... great, we get to make one!
            # two things, attempt to read from the watermark file.
            wm = make_watermark(length,prefix=kwargs["prefix"])




        target_image = YIQ_DCT_Image.open(f)
        mark = Marker(target=target_image,original=args.base,alpha=alpha)
        
        mark.embed(watermark=wm)
        result = mark.output()

        # check if the output file is good...
        #print(output_path)
        if (os.path.lexists(output_path)):
            if ((args.force) or confirm('Are you sure you want to overwrite {0}'.format(output_path))):
                print("Overwriting...")
                try:
                    result.write(output_path)
                except IOError as e:
                    exit(str(e))
        else:
            try:
                result.write(output_path)
            except IOError as e:
                exit(e)

        
        
                

        # at this point the watermarked image should be written already.
        # next up is storing the watermark.
        # if it already exists and we want a new mark...
        if (os.path.lexists(mark_path) and args.createnew):
            if ((args.force) or confirm('Are you sure you want to overwrite {0}'.format(mark_path))):
                print("Overwriting...")
                try:
                    write_watermark(wm,mark_path,file_base=base_path, file_target=output_path,alpha=args.alpha)
                except IOError as e:
                    exit(str(e))
        else: # if it doesn't make it.
            try:
                write_watermark(wm,mark_path,file_base=base_path, file_target=output_path,alpha=args.alpha)
            except IOError as e:
                exit(e)


#    raise NotImplementedError("Embed is not yet implemented.")
# parser_test.add_argument("-m", "--mark", help="Path to the watermark data. By default the name of the base file is also acquired from this.")
# parser_test.add_argument("-b", "--base", help="Base file for the coefficients",default=None)
# parser_test.add_argument("file", help="Path to the image file(s).", default=None)


def test(args,**kwargs):    
    from cox import Tester, YIQ_DCT_Image

    wmdata = read_watermark(args.mark,type="JSON")
    length = wmdata["length"]
    if (args.length != -1):
        v = int(args.length)
        if (v > length):
            exit("Desired length is longer then whats available")
        length = v
    alpha = wmdata["alpha"]
    base_path = wmdata["base"]

    if (args.base != None):
        base_path = args.base
    print(base_path)

    #files = args.file
    #print(args.file)
    files = args.file
    if (len(files) == 0):
        files.append(str(wmdata["target"]))
    
    base_image = YIQ_DCT_Image.open(str(base_path))


    for f in files:
        tester = Tester(f, base_image, alpha=alpha, length=length)
        res = tester.test(wmdata["wm"][0:length])
        #print(res)
        if (res[0]):
            print("Positive: file %s -> (%f,%f)"%(f,res[1][0],res[1][1]))
        else:
            print("Negative: file %s -> (%f,%f)"%(f,res[1][0],res[1][1]))

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
parser_embed.add_argument("-a", "--alpha", help="Alpha, strength of watermark, as ratio, default:" + str(default_alpha), default=default_alpha, type=float)
parser_embed.add_argument("-l", "--length", help="Length of watermark, defaults to "+ str(default_length)+ ". Can be specified with a %% in which case a percentage of the number of pixels will be used, or as a number, in which case the length of the watermark will be equal to this number.", default=default_length)


# We either use a watermark file, or we create one with a default name.
# if we should create one and the mark file is set we write it to the mark file.
#group = parser_embed.add_mutually_exclusive_group()
parser_embed.add_argument("-m", "--mark", metavar="<Wm File>",
                help="Path to the watermark data. Defaults to: "+ default_watermark_file, default=default_watermark_file)
parser_embed.add_argument("-c", "--createnew", action="store_true",help="Create a new watermark and save it to the <Wm File>.", default=False)

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
parser_test.add_argument("-m", "--mark", help="Path to the watermark data. By default the name of the base file is also acquired from this.")
parser_test.add_argument("-l", "--length", help="Maximum length of watermark, by default uses the total length in the watermarking file", default=-1)
parser_test.add_argument("-b", "--base", help="Base file for the coefficients",default=None)
parser_test.add_argument("file", help="Path to the image file(s).", default=None, nargs='*')





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

