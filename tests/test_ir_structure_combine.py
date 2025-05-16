import unittest
import numpy as np
from irs.ir_structure import gaussian, reconstruct_spectrum, combine_spectra_from_peaks

# --- Unit tests ---
class TestIRSpectrumPipeline(unittest.TestCase):
    def test_gaussian_peak_center(self):
        x = np.linspace(0, 10, 1000)
        peak = gaussian(x, 5, 1, 1)
        max_idx = np.argmax(peak)
        self.assertAlmostEqual(x[max_idx], 5, places=2)

    def test_gaussian_intensity_height(self):
        x = np.linspace(0, 10, 1000)
        g = gaussian(x, 5, 2, 1)
        self.assertAlmostEqual(np.max(g), 2, delta=0.01)

    def test_reconstruct_spectrum_multiple_peaks(self):
        x = np.linspace(0, 10, 1000)
        peaks = [(3, 1, 0.5), (7, 1, 0.5)]
        spectrum = reconstruct_spectrum(x, peaks)
        self.assertEqual(len(spectrum), len(x))
        self.assertAlmostEqual(np.max(spectrum), 1, delta=0.1)

    def test_combine_spectra_shape_and_bounds(self):
        x = np.linspace(400, 4000, 5000)
        test_dict = {
            "A": [(1000, 1, 30)],
            "B": [(3000, 0.5, 50)]
        }
        xx, yy = combine_spectra_from_peaks(test_dict, ["A", "B"], common_axis=x)
        self.assertEqual(len(xx), 5000)
        self.assertEqual(len(yy), 5000)
        self.assertTrue(np.all(yy <= 1.0))
        self.assertTrue(np.all(yy >= 0.0))

if __name__ == "__main__":
    unittest.main()














