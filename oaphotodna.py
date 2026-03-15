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

# ----- Helper -----


def clamp(val, min_, max_):
    return max(min_, min(max_, val))


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

# This parameter is used to switch between "robust" and "short"
# hashes. It is not clear how exactly this is intended to be used
# (e.g. "short" hashes have a totally different postprocessing step).
# The only value used in practice is 6. Changing it may or may not work.
GRID_SIZE_HYPERPARAMETER = 6


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


# ----- (3.2) Feature extraction -----

# This is equal to 26. This means that the `u` and `v` coordinates
# mentioned in the paper both range from [0, 25].
FEATURE_GRID_DIM = GRID_SIZE_HYPERPARAMETER * 4 + 2

# This is used to compute the step size which maps
# from grid points to image points. (It is not the step size itself.)
# This is slightly bigger than the feature grid dimensions in order to
# make each region overlap slightly.
FEATURE_STEP_DIVISOR = GRID_SIZE_HYPERPARAMETER * 4 + 4


# This is Equation 9 in the paper. It performs bilinear interpolation.
# The purpose of this is to better approximate the pixel information
# at a coordinate which is not an integer (and thus lies *between* pixels).
def interpolate_px_quad(summed_im, im_w, x, y, x_residue, y_residue, debug_str=''):
    px_1 = summed_im[y * im_w + x]
    px_2 = summed_im[(y+1) * im_w + x]
    px_3 = summed_im[y * im_w + x + 1]
    px_4 = summed_im[(y+1) * im_w + x + 1]
    # NOTE: Must multiply the interpolation factors first *and then* the pixel
    # (due to rounding behavior)
    px_avg = \
        ((1 - x_residue) * (1 - y_residue) * px_1) + \
        ((1 - x_residue) * y_residue * px_2) + \
        (x_residue * (1 - y_residue) * px_3) + \
        (x_residue * y_residue * px_4)
    if DEBUG_LOGGING:
        print(f"px {debug_str} {px_1} {px_2} {px_3} {px_4} | {px_avg}")
    return px_avg


# This eventually computes Equation 10 in the paper.
# This "box sum" is a blurred average over regions of the image.
def box_sum_for_radius(
        summed_im, im_w, im_h,
        grid_step_h, grid_step_v,
        grid_point_x, grid_point_y,
        radius, weight):

    # Compute where the corners are. This is Equation 6.
    corner_a_x = grid_point_x - radius * grid_step_h - 1
    corner_a_y = grid_point_y - radius * grid_step_v - 1
    corner_d_x = grid_point_x + radius * grid_step_h
    corner_d_y = grid_point_y + radius * grid_step_v
    # Make sure the corners are within the image bounds
    corner_a_x = clamp(corner_a_x, 0, im_w - 2)
    corner_a_y = clamp(corner_a_y, 0, im_h - 2)
    if DEBUG_LOGGING:
        print(f"corner r{radius} {corner_a_x} {corner_a_y} | {corner_d_x} {corner_d_y}")

    # Get an integer pixel coordinate for the corners.
    # This is Equation 7.
    corner_a_x_int = int(corner_a_x)
    corner_a_y_int = int(corner_a_y)
    corner_d_x_int = int(corner_d_x)
    corner_d_y_int = int(corner_d_y)
    # Compute the fractional part, since we need it for interpolation.
    # This is Equation 8.
    corner_a_x_residue = corner_a_x - corner_a_x_int
    corner_a_y_residue = corner_a_y - corner_a_y_int
    corner_d_x_residue = corner_d_x - corner_d_x_int
    corner_d_y_residue = corner_d_y - corner_d_y_int
    if DEBUG_LOGGING:
        print(f"corner int r{radius} {corner_a_x_int} {corner_a_y_int} | {corner_d_x_int} {corner_d_y_int}")

    # Fetch the pixels in each corner
    px_A = interpolate_px_quad(
        summed_im,
        im_w,
        corner_a_x_int,
        corner_a_y_int,
        corner_a_x_residue,
        corner_a_y_residue,
        f"r{radius} A")
    px_B = interpolate_px_quad(
        summed_im,
        im_w,
        corner_d_x_int,
        corner_a_y_int,
        corner_d_x_residue,
        corner_a_y_residue,
        f"r{radius} B")
    px_C = interpolate_px_quad(
        summed_im,
        im_w,
        corner_a_x_int,
        corner_d_y_int,
        corner_a_x_residue,
        corner_d_y_residue,
        f"r{radius} C")
    px_D = interpolate_px_quad(
        summed_im,
        im_w,
        corner_d_x_int,
        corner_d_y_int,
        corner_d_x_residue,
        corner_d_y_residue,
        f"r{radius} D")

    # Compute the final sum. This is Equation 10 and 11, rearranged.
    # NOTE: The computation needs to be performed like this for rounding to match.
    R_box = px_A * weight - px_B * weight - px_C * weight + px_D * weight
    if DEBUG_LOGGING:
        print(f"box sum r{radius} {R_box}")
    return R_box


def compute_feature_grid(summed_im, im_w, im_h):
    # Compute the grid step size, which is Delta_l and Delta_w in the paper.
    # The paper does not explain how to do this.
    grid_step_h = im_w / FEATURE_STEP_DIVISOR
    grid_step_v = im_h / FEATURE_STEP_DIVISOR
    if DEBUG_LOGGING:
        print(f"step {grid_step_h} {grid_step_v}")

    feature_grid = [0.0] * (FEATURE_GRID_DIM * FEATURE_GRID_DIM)
    for feat_y in range(FEATURE_GRID_DIM):
        for feat_x in range(FEATURE_GRID_DIM):
            if DEBUG_LOGGING:
                print(f"-- grid {feat_x} {feat_y} --")

            # Compute what pixel the feature grid point maps to in the source image.
            # This is Equation 5 in the paper. The value of zeta is 1.5.
            grid_point_x = (feat_x + 1.5) * grid_step_h
            grid_point_y = (feat_y + 1.5) * grid_step_v
            if DEBUG_LOGGING:
                print(f"grid point {grid_point_x} {grid_point_y}")

            # Compute the box sum for each radius.
            # The radii scaling factors are 0.2, 0.4, and 0.8.
            radius_box_0p2 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.2, WEIGHT_R1)
            radius_box_0p4 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.4, WEIGHT_R2)
            radius_box_0p8 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.8, WEIGHT_R3)

            # Compute the final feature value. This is Equation 11.
            # See NOTE about rounding within `box_sum_for_radius`
            feat_val = radius_box_0p2 + radius_box_0p4 + radius_box_0p8
            if DEBUG_LOGGING:
                print(f"--> {feat_val}")
            feature_grid[feat_y * FEATURE_GRID_DIM + feat_x] = feat_val

    return feature_grid

# ----- Put it all together -----


def compute_hash(filename):
    # Load image
    im = Image.open(filename)
    # Preprocess into summed array
    if not USE_NUMPY:
        summed_pixels = preprocess_pixel_sum(im)
    else:
        summed_pixels = preprocess_pixel_sum_np(im)
    feature_grid = compute_feature_grid(summed_pixels, im.width, im.height)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} filename")
        sys.exit(-1)
    compute_hash(sys.argv[1])
