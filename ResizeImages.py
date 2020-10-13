
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def test_directory(d):
    if ( not os.path.isdir(d) ):
        os.makedirs(d)

def handle_arguments():
    parser = argparse.ArgumentParser(description='Resize images found in a directory. ')

    parser.add_argument('indir', type=str,
        help='The input directory. ')

    parser.add_argument('outdir', type=str,
        help='The output directory. ')

    parser.add_argument('height', type=int,
        help='The height of the resized image. ')

    parser.add_argument('width', type=int,
        help='The width of the resized image. ' )

    parser.add_argument('--pattern', type=str, default='*.png',
        help='The search pattern. ')

    args = parser.parse_args()

    return args

def find_files(d, pattern):
    ss = '%s/**/%s' % (d, pattern)
    files = sorted( glob.glob( ss, recursive=True ) )

    if ( 0 == len(files) ):
        raise Exception('No files found with %s. ' % (ss))

    return files

def get_target_filename(outdir, fn):
    '''
    outdir (string): The output direcotry.
    fn (string): The input file.
    '''

    split0 = os.path.split(fn)
    split1 = os.path.split(split0[0])

    prefix = os.path.join( outdir, split1[1])

    test_directory(prefix)

    p = os.path.join(prefix, split0[1])

    return p

def load_and_resize_img(fn, shape):
    '''
    fn (string): The image filename.
    shape (2-element array): (H, W). The new shape.
    '''

    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist. ' % (fn))

    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    return img

def main():
    print('Hello, Analysis! ')

    args = handle_arguments()

    test_directory(args.outdir)

    files = find_files(args.indir, args.pattern)

    for f in files:
        print(f)

        img = load_and_resize_img(f, ( args.height, args.width ))
        outFn = get_target_filename(args.outdir, f)

        cv2.imwrite(outFn, img)

        print(outFn)

    return 0

if __name__ == '__main__':
    sys.exit( main() )

