"""Day 7 tests for explicit CUDA host/device memory transfers."""
import unittest
from unittest import mock

import numpy as np

import fasta as f


class HostDeviceTransfers(unittest.TestCase):
    def test_host_to_device_makes_noncontiguous_input_contiguous(self):
        source = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
        fake_cuda = mock.Mock()
        fake_cuda.to_device.side_effect = lambda array, stream=0: (array, stream)

        with mock.patch.object(f, "HAVE_NUMBA", True), mock.patch.object(f, "cuda", fake_cuda):
            copied, stream = f.host_to_device(source, stream=3)

        np.testing.assert_array_equal(copied, source)
        self.assertTrue(copied.flags.c_contiguous)
        self.assertEqual(stream, 3)
        fake_cuda.to_device.assert_called_once()

    def test_host_to_device_rejects_object_arrays(self):
        with mock.patch.object(f, "HAVE_NUMBA", True):
            with self.assertRaisesRegex(TypeError, "fixed-size dtype"):
                f.host_to_device(np.array([object()], dtype=object))

    def test_host_to_device_requires_numba(self):
        with mock.patch.object(f, "HAVE_NUMBA", False):
            with self.assertRaisesRegex(RuntimeError, "require Numba CUDA"):
                f.host_to_device(np.arange(4, dtype=np.uint8))

    def test_device_to_host_uses_requested_stream(self):
        expected = np.arange(4, dtype=np.uint8)
        device_array = mock.Mock()
        device_array.copy_to_host.return_value = expected

        with mock.patch.object(f, "HAVE_NUMBA", True):
            result = f.device_to_host(device_array, stream=5)

        np.testing.assert_array_equal(result, expected)
        device_array.copy_to_host.assert_called_once_with(stream=5)

    def test_device_to_host_rejects_non_device_arrays(self):
        with mock.patch.object(f, "HAVE_NUMBA", True):
            with self.assertRaisesRegex(TypeError, "copy_to_host"):
                f.device_to_host(np.arange(4, dtype=np.uint8))

    def test_gpu_scan_uses_explicit_host_device_transfers(self):
        codes = f.encode_bases("ACGTACGTTGG")
        guides = np.array([f.encode_guide("ACGTACGT")], dtype=np.uint8)
        masks = f.pam_masks_for("NGG")
        fake_kernel = mock.MagicMock()
        h2d = mock.Mock(side_effect=lambda array: array)
        d2h = mock.Mock(side_effect=lambda array: array)

        with (mock.patch.object(f, "gpu_backend", return_value="gpu"),
              mock.patch.object(f, "host_to_device", h2d),
              mock.patch.object(f, "device_to_host", d2h),
              mock.patch.object(f, "mismatch_kernel", fake_kernel)):
            result = f.scan_codes(codes, guides, 8, masks, 0, backend="gpu")

        self.assertIs(result, h2d.call_args_list[-1].args[0])
        self.assertEqual(h2d.call_count, 4)
        self.assertTrue(np.all(h2d.call_args_list[-1].args[0] == 255))
        d2h.assert_called_once_with(result)
        fake_kernel.__getitem__.return_value.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
