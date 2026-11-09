#
# python_grabber
#
# Authors:
#  Andrea Schiavinato <andrea.schiavinato84@gmail.com>
#
# Copyright (C) 2019 Andrea Schiavinato
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

"""
NumPy-only frame filters for the SampleGrabber callback path.

The functions in this module deliberately depend on nothing but NumPy, so they
can be exercised off the DirectShow stream thread and unit-tested without any
camera. DirectShow delivers uncompressed frames as bottom-up BGR buffers
(RGB24 / RGB32) whose rows are padded to a 4-byte (DWORD) boundary; the
padding bytes must never take part in the computation.
"""

from __future__ import annotations

import numpy as np

# ITU-R BT.601 luma coefficients in B, G, R byte order. Scaled by 1000 so the
# weighted sum can be carried out in plain integer arithmetic: this keeps the
# channel order explicit, avoids uint8 overflow and gives deterministic,
# rounding-free float behavior. The weights sum to 1000.
BGR_LUMA_WEIGHTS = (0.114, 0.587, 0.299)
_WEIGHT_B, _WEIGHT_G, _WEIGHT_R = (114, 587, 299)
_WEIGHT_SCALE = 1000

_SUPPORTED_BITS_PER_PIXEL = (24, 32)


def row_stride(width: int, bits_per_pixel: int) -> int:
    """Return the DWORD-aligned row stride, in bytes, of a DIB frame.

    Each scanline occupies ``width * bits_per_pixel / 8`` bytes rounded up to
    the next multiple of four. An RGB24 frame whose width is not a multiple of
    four therefore carries padding bytes at the end of every row; RGB32 rows
    are naturally aligned.
    """
    if bits_per_pixel not in _SUPPORTED_BITS_PER_PIXEL:
        raise ValueError(f"Unsupported bit depth {bits_per_pixel}; expected 24 or 32")
    bytes_per_row = width * (bits_per_pixel // 8)
    return (bytes_per_row + 3) & ~3


def to_grayscale(frame: np.ndarray, width: int, height: int,
                 bits_per_pixel: int = 24, bottom_up: bool = True) -> np.ndarray:
    """Convert a native BGR24/BGR32 buffer to a top-down, 3-channel grayscale image.

    The input is the raw DirectShow buffer: a uint8 ndarray that is either flat
    (one element per buffer byte) or already reshaped to ``(height, stride)``.
    Rows are assumed bottom-up and padded to a 4-byte stride. Only the valid
    pixels are read; stride padding is excluded from the computation and the
    input buffer is never written to.

    The weighted luma is accumulated in int32 so uint8 values cannot overflow,
    rounded back into the 0..255 range and broadcast into a freshly allocated
    ``(height, width, 3)`` uint8 image. Keeping three channels means the
    recording and display code paths need no changes.

    Parameters
    ----------
    frame: np.ndarray
        uint8 buffer, 1-D of length ``height * stride`` or 2-D of shape
        ``(height, stride)``.
    width, height: int
        Visible frame dimensions in pixels.
    bits_per_pixel: int
        24 for RGB24 (BGR, 3 bytes/pixel) or 32 for RGB32 (BGRx, 4 bytes/pixel).
    bottom_up: bool
        True for the default DirectShow DIB layout; pass False for top-down input.
    """
    if bits_per_pixel not in _SUPPORTED_BITS_PER_PIXEL:
        raise ValueError(f"Unsupported bit depth {bits_per_pixel}; expected 24 or 32")

    channels = bits_per_pixel // 8
    stride = row_stride(width, bits_per_pixel)
    expected_size = height * stride

    buffer = np.asarray(frame, dtype=np.uint8)
    if buffer.ndim == 1:
        if buffer.size < expected_size:
            raise ValueError(f"Buffer of {buffer.size} bytes is too small for "
                             f"{width}x{height} at {bits_per_pixel} bpp (need {expected_size})")
        rows = buffer[:expected_size].reshape(height, stride)
    elif buffer.ndim == 2:
        if buffer.shape != (height, stride):
            raise ValueError(f"Expected a (height, stride) = ({height}, {stride}) array, "
                             f"got {buffer.shape}")
        rows = buffer
    else:
        raise ValueError(f"Expected a 1-D or 2-D uint8 buffer, got {buffer.ndim}-D")

    # Strided views over the valid bytes only: no reshape copy, and the row
    # padding columns are never indexed. For RGB32 the fourth (x) byte is
    # skipped together with any trailing stride padding.
    row_bytes = width * channels
    b = rows[:, 0:row_bytes:channels]
    g = rows[:, 1:row_bytes:channels]
    r = rows[:, 2:row_bytes:channels]

    # Accumulate in a wider integer type: a plain uint8 sum would wrap past 255.
    luma = (b.astype(np.int32) * _WEIGHT_B
            + g.astype(np.int32) * _WEIGHT_G
            + r.astype(np.int32) * _WEIGHT_R)
    luma = (luma + _WEIGHT_SCALE // 2) // _WEIGHT_SCALE

    if bottom_up:
        luma = luma[::-1, :]

    # Single output allocation: broadcast the luma plane onto all three BGR
    # channels in one contiguous copy.
    output = np.empty((height, width, 3), dtype=np.uint8)
    output[:] = luma[:, :, np.newaxis]
    return output
