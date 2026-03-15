#!/usr/bin/env python3

# ----- Import libraries, global settings -----

from math import floor, sqrt
from PIL import Image

DEBUG_LOGGING = True

if DEBUG_LOGGING:
    import struct
try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

# ----- Extracted constants -----

# These constants are used as weights for each differently-sized
# rectangle during the feature extraction phase.
# This is used in Equation 11 in the paper.
WEIGHT_R1 = float.fromhex('0x1.936398bf0aae3p-3')
WEIGHT_R2 = float.fromhex('0x1.caddcd96f4881p-2')
WEIGHT_R3 = float.fromhex('0x1.1cb5cf620ef1dp-2')

# This is used for initial hash scaling.
# This is described in section 3.4 of the paper.
HASH_SCALE_CONST = float.fromhex('0x1.07b3705abb25cp0')


# ----- (3.1) Preprocessing -----

# Compute the summed pixel data. The summed data has the same dimensions
# as the input image. For each pixel position, the output at that point
# is the sum of all pixels in the rectangle from the origin over to
# the given point. The RGB channels are summed together.
def preprocess_pixel_sum(im):
    sum_out = []

    # The first row does not have a row above it, so we treat it specially
    accum = 0
    for x in range(im.width):
        px = im.getpixel((x, 0))
        # Sum RGB channels
        pxsum = px[0] + px[1] + px[2]
        # As the x coordinate moves right, we sum up everything
        # starting from the beginning of the row.
        accum += pxsum
        sum_out.append(accum)

    # For all subsequent rows, there is a row above.
    # We can save a lot of processing time by reusing that information.
    # (This is a straightforward example of "dynamic programming".)
    for y in range(1, im.height):
        accum = 0
        for x in range(im.width):
            px = im.getpixel((x, y))
            # Sum RGB channels
            pxsum = px[0] + px[1] + px[2]
            # `accum` is the sum of just this row
            accum += pxsum
            # Re-use already-computed data from previous row
            last_row_sum = sum_out[(y-1) * im.width + x]
            sum_out.append(accum + last_row_sum)

    return sum_out


# Optimized implementation using NumPy
def preprocess_pixel_sum_np(im):
    # Convert to NumPy
    im = np.array(im, dtype=np.uint64)
    # Sum RGB components
    im = im.sum(axis=2)
    # Sum along each row ("over" columns)
    im = im.cumsum(axis=1)
    # Sum down the image ("over" rows)
    im = im.cumsum(axis=0)
    return im.flatten()


# ----- Put it all together -----


def compute_hash(filename):
    # Load image
    im = Image.open(filename)
    # Preprocess into summed array
    if not USE_NUMPY:
        summed_pixels = preprocess_pixel_sum(im)
    else:
        summed_pixels = preprocess_pixel_sum_np(im)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} filename")
        sys.exit(-1)
    compute_hash(sys.argv[1])
