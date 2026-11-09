# Tests for the SampleGrabber callback path: simulated DirectShow buffers
# delivered through SampleGrabberCallback.BufferCB, grayscale toggle behavior,
# and flag propagation through FilterGraph / PyGrabber without any camera.

import ctypes

import numpy as np
import pytest

from pygrabber.PyGrabber import PyGrabber
from pygrabber.dshow_graph import FilterGraph, FilterType, SampleGrabberCallback
from pygrabber.filters import row_stride, to_grayscale

PAD_SENTINEL = 0xCD


def make_native_buffer(width, height, bpp, pad_value=PAD_SENTINEL):
    """Allocate a ctypes byte buffer shaped like a DirectShow DIB frame."""
    channels = bpp // 8
    stride = row_stride(width, bpp)
    size = height * stride
    native = (ctypes.c_ubyte * size)()
    view = np.ctypeslib.as_array(native, shape=(size,))
    view[:] = pad_value
    rows = view.reshape(height, stride)
    for row in range(height):
        for col in range(width):
            for ch in range(3):
                rows[row, col * channels + ch] = (row * 17 + col * 5 + ch * 13 + 1) % 256
    if channels == 4:
        rows[:, 3::4] = pad_value
    return native, view, rows, stride, size


def make_callback(width, height, bpp):
    received = []
    cb = SampleGrabberCallback(lambda img: received.append(img))
    cb.image_resolution = (width, height)
    cb.image_bpp = bpp
    return cb, received


def fire(cb, native, size):
    cb.grab_frame()
    result = cb.BufferCB(None, 0, native, size)
    assert result == 0


def test_callback_rgb24_color_path_respects_stride_and_flip():
    width, height, bpp = 5, 3, 24
    native, view, rows, stride, size = make_native_buffer(width, height, bpp)
    assert stride == 16  # 15 valid bytes + 1 padding byte per row
    snapshot = view.copy()

    cb, received = make_callback(width, height, bpp)
    fire(cb, native, size)

    assert len(received) == 1
    img = received[0]
    assert img.shape == (height, width, 3) and img.dtype == np.uint8
    # Bottom-up: first output row comes from the last buffer row.
    expected = np.flip(rows[:, :width * 3].reshape(height, width, 3), axis=0)
    assert np.array_equal(img, expected)
    # Native buffer, padding included, is left untouched.
    assert np.array_equal(view, snapshot)
    assert np.all(rows[:, width * 3:] == PAD_SENTINEL)


def test_callback_rgb24_grayscale_path_matches_filter():
    width, height, bpp = 7, 4, 24  # 21 valid bytes, padded to 24
    native, view, rows, stride, size = make_native_buffer(width, height, bpp)
    snapshot = view.copy()

    cb, received = make_callback(width, height, bpp)
    cb.grayscale_enabled = True
    fire(cb, native, size)

    img = received[0]
    assert img.shape == (height, width, 3)
    expected = to_grayscale(rows, width, height, bpp)
    assert np.array_equal(img, expected)
    # All three channels hold the same luma value.
    assert np.array_equal(img[:, :, 0], img[:, :, 1])
    assert np.array_equal(img[:, :, 1], img[:, :, 2])
    assert np.array_equal(view, snapshot)
    assert np.all(rows[:, width * 3:] == PAD_SENTINEL)


def test_callback_rgb32_drops_fourth_byte():
    width, height, bpp = 3, 2, 32
    native, view, rows, stride, size = make_native_buffer(width, height, bpp)

    cb, received = make_callback(width, height, bpp)
    fire(cb, native, size)
    img = received[0]
    assert img.shape == (height, width, 3)
    expected = np.flip(rows[:, :width * 4].reshape(height, width, 4)[:, :, :3], axis=0)
    assert np.array_equal(img, expected)

    cb.grayscale_enabled = True
    fire(cb, native, size)
    assert received[1].shape == (height, width, 3)
    assert np.array_equal(received[1], to_grayscale(rows, width, height, bpp))


def test_toggle_takes_effect_on_next_frame_without_rebuild():
    width, height, bpp = 5, 2, 24
    native, view, rows, stride, size = make_native_buffer(width, height, bpp)
    cb, received = make_callback(width, height, bpp)

    # Frames before toggling: genuine color, channels differ somewhere.
    fire(cb, native, size)
    fire(cb, native, size)
    for img in received:
        assert not np.array_equal(img[:, :, 0], img[:, :, 2])

    # Flip mid-stream: the next frame is fully grayscale (no half-converted frame).
    cb.grayscale_enabled = True
    fire(cb, native, size)
    gray = received[-1]
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1])
    assert np.array_equal(gray[:, :, 1], gray[:, :, 2])
    assert np.array_equal(gray, to_grayscale(rows, width, height, bpp))

    # Flip back: color frames resume.
    cb.grayscale_enabled = False
    fire(cb, native, size)
    assert not np.array_equal(received[-1][:, :, 0], received[-1][:, :, 2])


def test_grab_frame_flag_consumed_once():
    width = height = 1
    native, _, _, _, size = make_native_buffer(width, height, 24)
    cb, received = make_callback(width, height, 24)

    assert cb.keep_photo is False
    cb.grab_frame()
    assert cb.keep_photo is True
    cb.BufferCB(None, 0, native, size)
    assert cb.keep_photo is False and len(received) == 1
    # A further buffer without grab_frame() must not reach the user callback.
    cb.BufferCB(None, 0, native, size)
    assert len(received) == 1


def test_filter_graph_propagates_and_persists_grayscale_flag():
    graph = FilterGraph()
    assert graph.is_grayscale_enabled() is False

    graph.add_sample_grabber(lambda img: None)
    callback = graph.filters[FilterType.sample_grabber].callback
    assert callback.grayscale_enabled is False

    graph.set_grayscale(True)  # live toggle on the existing callback
    assert graph.is_grayscale_enabled() is True
    assert callback.grayscale_enabled is True

    # Simulate the preview -> recording rebuild: filters are removed and a new
    # SampleGrabber is created; it must inherit the graph-level state.
    graph.remove_filters()
    graph.add_sample_grabber(lambda img: None)
    new_callback = graph.filters[FilterType.sample_grabber].callback
    assert new_callback is not callback
    assert new_callback.grayscale_enabled is True

    graph.set_grayscale(False)
    assert new_callback.grayscale_enabled is False


def test_pygrabber_exposes_set_grayscale():
    grabber = PyGrabber(lambda img: None)
    assert grabber.graph.is_grayscale_enabled() is False
    grabber.set_grayscale(True)
    assert grabber.graph.is_grayscale_enabled() is True
    grabber.set_grayscale(False)
    assert grabber.graph.is_grayscale_enabled() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
