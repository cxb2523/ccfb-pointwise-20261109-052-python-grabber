# Tests for the NumPy-only grayscale filter: odd-width RGB24/RGB32 buffers
# with stride padding, luma values, shape/channels, buffer preservation.

import numpy as np
import pytest

from pygrabber.filters import BGR_LUMA_WEIGHTS, row_stride, to_grayscale

PAD_SENTINEL = 0xAB


def make_bottom_up_buffer(width, height, bits_per_pixel, fill=None, pad_value=PAD_SENTINEL):
    """Build a native, bottom-up, DWORD-padded BGR buffer and its expected layout."""
    channels = bits_per_pixel // 8
    stride = row_stride(width, bits_per_pixel)
    buffer = np.full(height * stride, pad_value, dtype=np.uint8)
    rows = buffer.reshape(height, stride)
    if fill is None:
        # Deterministic pixel pattern depending on row/column/channel.
        for row in range(height):
            for col in range(width):
                for ch in range(3):
                    rows[row, col * channels + ch] = (row * 31 + col * 7 + ch * 11) % 256
    else:
        for row in range(height):
            for col in range(width):
                rows[row, col * channels:col * channels + 3] = fill
        if channels == 4:
            rows[:, 3::4] = pad_value
    return buffer, rows, stride


def expected_luma(b, g, r):
    weights = BGR_LUMA_WEIGHTS
    return int(round(float(b) * weights[0] + float(g) * weights[1] + float(r) * weights[2]))


def test_row_stride_dword_alignment():
    assert row_stride(4, 24) == 12       # 4 * 3, already aligned
    assert row_stride(5, 24) == 16       # 15 bytes -> padded to 16 (odd width)
    assert row_stride(7, 24) == 24       # 21 bytes -> padded to 24
    assert row_stride(5, 32) == 20       # RGB32 rows are naturally aligned
    assert row_stride(3, 32) == 12
    with pytest.raises(ValueError):
        row_stride(5, 16)


def test_rgb24_odd_width_shape_channels_and_values():
    width, height, bpp = 5, 3, 24
    buffer, rows, stride = make_bottom_up_buffer(width, height, bpp)
    assert stride == width * 3 + 1       # one padding byte per row

    result = to_grayscale(buffer, width, height, bpp)

    assert result.dtype == np.uint8
    assert result.shape == (height, width, 3)
    # Bottom-up flip: output top row is the last buffer row.
    for out_row in range(height):
        buffer_row = height - 1 - out_row
        for col in range(width):
            b, g, r = (int(rows[buffer_row, col * 3 + ch]) for ch in range(3))
            expected = expected_luma(b, g, r)
            assert tuple(result[out_row, col]) == (expected, expected, expected)


def test_rgb24_padding_bytes_excluded_and_untouched():
    width, height, bpp = 5, 2, 24
    buffer, rows, stride = make_bottom_up_buffer(width, height, bpp)
    before = buffer.copy()

    result = to_grayscale(buffer, width, height, bpp)

    # Input buffer must not be modified at all, padding included.
    assert np.array_equal(buffer, before)
    assert np.all(rows[:, width * 3:] == PAD_SENTINEL)
    # Output has no padding columns: exactly width pixels per row.
    assert result.shape[1] == width


def test_rgb32_fourth_byte_and_padding_ignored():
    width, height, bpp = 3, 2, 32
    stride = row_stride(width, bpp)
    assert stride == 12 and width * 4 == 12

    buffer_a = np.zeros(height * stride, dtype=np.uint8)
    buffer_b = np.zeros(height * stride, dtype=np.uint8)
    for buf, fourth_byte in ((buffer_a, 0), (buffer_b, 255)):
        rows = buf.reshape(height, stride)
        for row in range(height):
            for col in range(width):
                rows[row, col * 4:col * 4 + 3] = (10, 128, 200)
            rows[row, 3::4] = fourth_byte

    result_a = to_grayscale(buffer_a, width, height, bpp)
    result_b = to_grayscale(buffer_b, width, height, bpp)
    assert np.array_equal(result_a, result_b)          # x byte never participates
    assert np.all(buffer_b.reshape(height, stride)[:, 3::4] == 255)
    assert result_a.shape == (height, width, 3)


def test_uint8_overflow_is_avoided():
    width = height = 1
    buffer, _, _ = make_bottom_up_buffer(width, height, 24, fill=(255, 255, 255))
    result = to_grayscale(buffer, width, height, 24)
    assert tuple(result[0, 0]) == (255, 255, 255)      # would wrap if summed in uint8

    buffer, _, _ = make_bottom_up_buffer(width, height, 24, fill=(0, 0, 0))
    result = to_grayscale(buffer, width, height, 24)
    assert tuple(result[0, 0]) == (0, 0, 0)


def test_bgr_channel_order_not_reversed():
    # If B and R were swapped, blue and red would exchange their luma values.
    blue, _, _ = make_bottom_up_buffer(1, 1, 24, fill=(255, 0, 0))
    green, _, _ = make_bottom_up_buffer(1, 1, 24, fill=(0, 255, 0))
    red, _, _ = make_bottom_up_buffer(1, 1, 24, fill=(0, 0, 255))
    assert to_grayscale(blue, 1, 1, 24)[0, 0, 0] == expected_luma(255, 0, 0)
    assert to_grayscale(green, 1, 1, 24)[0, 0, 0] == expected_luma(0, 255, 0)
    assert to_grayscale(red, 1, 1, 24)[0, 0, 0] == expected_luma(0, 0, 255)
    assert to_grayscale(blue, 1, 1, 24)[0, 0, 0] != to_grayscale(red, 1, 1, 24)[0, 0, 0]


def test_top_down_input_and_2d_equivalence():
    width, height, bpp = 5, 3, 24
    buffer, rows, stride = make_bottom_up_buffer(width, height, bpp)

    flat_result = to_grayscale(buffer, width, height, bpp, bottom_up=False)
    two_d_result = to_grayscale(rows.copy(), width, height, bpp, bottom_up=False)
    assert np.array_equal(flat_result, two_d_result)
    # No flip: output top row is the first buffer row.
    b, g, r = (int(rows[0, ch]) for ch in range(3))
    assert flat_result[0, 0, 0] == expected_luma(b, g, r)

    flipped = to_grayscale(buffer, width, height, bpp, bottom_up=True)
    assert np.array_equal(flipped[0], flat_result[-1])


def test_invalid_buffers_raise():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros(10, dtype=np.uint8), 5, 2, 24)   # too small
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 15), dtype=np.uint8), 5, 2, 24)  # wrong 2-D shape
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 5, 3), dtype=np.uint8), 5, 2, 24)  # 3-D rejected
