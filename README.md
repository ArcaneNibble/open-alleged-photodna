# Open Alleged PhotoDNA

This code is a complete and public implementation of _Alleged PhotoDNA_.

## Background

PhotoDNA is an algorithm owned by Microsoft which performs [perceptual hashing](https://en.wikipedia.org/wiki/Perceptual_hashing) of images. The purpose of a perceptual hash is to generate a _hash_ value which hopefully somehow meaningfully captures visual similarity. Small edits to an image such as cropping, resizing, or blanking out a small region should ideally result in either the same or a very similar hash value. This is in contrast to _cryptographic_ hashes which try to detect any changes whatsoever. Hashes are used because they are typically smaller than the original data and are also non-invertible (making it impossible to derive the original data from only the hash).

The PhotoDNA algorithm was developed to help identify child pornography/child sexual abuse material. However, it is a proprietary algorithm whose exact details are kept secret. This has limited the ability to study and analyze it.

Recently (March 2026), researchers reverse-engineered and described the algorithm and developed several attacks [[1]](#references). They also mention that a leaked PhotoDNA binary has been available on the Internet since 2021. However, this paper omits some details needed for a complete end-to-end implementation.

This code fills in the missing details in order to implement an open-source end-to-end _Alleged PhotoDNA_.

This code is meant to be read alongside the paper [[1]](#references). It is also helpful to read a blog post [[2]](#references) describing the algorithm in a different way.

The remainder of this README will contain various reverse-engineer's notes, implementation notes, and commentary.

## Setup

This code requires Pillow (to parse images). NumPy is optional but strongly recommended (to drastically speed up input preprocessing).

By default, the tool processes a single image passed as a command-line argument:

```sh
./oaphotodna.py image.jpg
```

Alternatively, two images can be passed in order to compare their hashes:

```sh
./oaphotodna.py image1.jpg image2.jpg
```

This code has also been validated against the leaked binary on the ImageNet 2012 validation set. There is commented-out code that checks this.

## Input preprocessing

All public code which calls the leaked binary uses the `ComputeRobustHash` function. The signature of this function is:

```c
int ComputeRobustHash(const void *input_bytes, uint w, uint h, uint stride, void *out, struct buffer *buf)
```

The input data to this function is expected in 24-bits-per-pixel RGB format.

The dimensions are specified in pixels.

The `stride` parameter is usually passed as `0` in order to make it equal to the image width.

The `buffer` parameter can be used to manually manage temporary memory. This can reduce heap traffic when performing batch operations. This parameter can be null in order to have it managed automatically.

There are other exported function calls with different and often more complicated functionality. Almost all of them eventually end up calling into a function which I have called `compute_hash_real` at `+0x2160`.

Notably, using the other functions it is possible to specify other pixel formats, and to do _something unknown_ with image borders.

There are 5 pixel formats, but it is not clear what the intended use case is. They have the following number of bytes per pixel: `3, 4, 1, 3, 1`. Speculation: RGB, RGBA, Grayscale, unknown, unknown. These pixel formats also affect the final hash scaling factor (line 382 in the code), with the following values: `3, 3, 1, 1, 3`.

If the image has a border or unexpected stride, some of that data is hashed with a SHA-1 hash, and _something unknown_ is done with it. This has not been reverse engineered. Wild speculation: this might be used by some calling tools in order to defend against the border attack described in the paper.

Initial processing is done in the function at `+0xe9c0`

## Resizing

The actual implementation in the leaked binary does not perform any image resizing, even though blog post [[2]](#references) describes such. Instead, sparse sampling of the _summed_ pixel data performs a conceptually equivalent computation.

## Feature extraction

The main processing is performed in the function at `+0x8610`.

The "feature grid" described in [[1]](#references) is the 26×26 pixels described in [[2]](#references). The "2-pixel overlap" is implemented by adding 2 when computing the grid step size.

The feature grid _and_ gradient grid are both stored on the stack. In order to test for bit-exact matching output, it is possible to set breakpoints at the appropriate location and dump these intermediate values for comparison.

At `+0x9902`, the feature grid is done being computed.

It is tricky to fully trace and annotate the computation, especially since the compiler seems to be very prone to overlapping operations (possibly for CPU pipeline optimization?). This code was primarily implemented based on the paper, and the binary was used only to check order of operations in order to get bit-exact matching.

In the left side of Fig. 2, the dots represent feature grid coordinates. Image pixels are not shown.

In the right side of Fig. 2, the light grey grid corresponds to image pixels. I am not sure why the coordinate rounding/truncation is depicted the way it is. Truncation always moves points towards the origin, and the interpolation is performed the exact same way for each corner A/B/C/D.

## Gradient processing

Note that 26×26 becomes 24×24 after removing 1 pixel from _each_ side top/bottom/left/right.

The binary processes 6×6 chunks of 4×4 values, but the innermost loop is unrolled. This is why there are 4 identical copies of the inner logic. The coordinates used in the binary match up with the _bottommost_ value of the `+`-shaped gradient operator, but this code here uses the center instead. This differs by 1 row.

Gradients are spread into the gradient grid using an "inverse" bilinear interpolation. See the code for details.

At `+0xc041`, the feature grid is done being computed.

## Hash normalization

The PhotoDNA algorithm has a hyperparameter for tuning the gradient and feature grid size. Normally this is 6 (and this is the upper limit for the binary), but it can also be 4 for "short" hashes. It is not known what this is actually used for, but modes other than `6` use a different hash post-processing algorithm. The description in the paper and the implementation in this code both expect the `6` mode.

The binary contains a huge mess of extra local variables stashing a copy of the work-in-progress hash vector. It's not clear why, but _this can be ignored_. The innermost loop of the hash normalization is unrolled, also creating a mess of code.

At `+0xe8ae`, all the iterations are completed, but the hash is not yet converted to bytes. A breakpoint can be set here to check for bit-exact matching.

## References

1. [White-Box Attacks on PhotoDNA Perceptual Hash Function](https://eprint.iacr.org/2026/486)
2. [PhotoDNA and Limitations](https://www.hackerfactor.com/blog/index.php?/archives/931-PhotoDNA-and-Limitations.html)
